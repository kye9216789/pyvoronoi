"""Algorithm and pipeline functions for pyvoronoi."""

from __future__ import annotations

import importlib
from typing import Optional, Sequence

import numpy as np

from pyvoronoi.model import Cell, Edge, Vertex, VoronoiException

_pyvoronoi = importlib.import_module("_pyvoronoi")
Point = _pyvoronoi.Point
Segment = _pyvoronoi.Segment
_VoronoiDiagram = _pyvoronoi.VoronoiDiagram
_generate_internal_segments_native = _pyvoronoi.generate_internal_segments
_generate_internal_graph_native = _pyvoronoi.generate_internal_graph

SILENT = True


def log_action(description: str) -> None:
    """Log a message when SILENT is False."""
    if not SILENT:
        print(description)


def _point_to_array(point) -> list[int]:
    """Convert a native Point to a [x, y] list."""
    return [point.X, point.Y]


def _segment_to_array(segment) -> list[list[int]]:
    """Convert a native Segment to a [[x0, y0], [x1, y1]] list."""
    return [[segment.p0.X, segment.p0.Y], [segment.p1.X, segment.p1.Y]]


def generate_internal_segments(exterior, scaling_factor: int = 1):
    """Build internal Voronoi segments from a polygon exterior."""
    exterior_array = np.asarray(exterior, dtype=np.float64)
    return _generate_internal_segments_native(exterior_array, scaling_factor)


def generate_internal_graph(exterior, scaling_factor: int = 1):
    """Build an sknw-style networkx Graph from a polygon exterior."""
    exterior_array = np.asarray(exterior, dtype=np.float64)
    graph_data = _generate_internal_graph_native(exterior_array, scaling_factor)
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError("networkx is required for generate_internal_graph") from exc

    graph = nx.Graph()
    nodes = np.asarray(graph_data["nodes"], dtype=np.float64)
    for node_id, node_yx in enumerate(nodes):
        graph.add_node(int(node_id), o=node_yx)

    edges_u = np.asarray(graph_data["edges_u"], dtype=np.int64)
    edges_v = np.asarray(graph_data["edges_v"], dtype=np.int64)
    offsets = np.asarray(graph_data["edge_offsets"], dtype=np.int64)
    edge_points = np.asarray(graph_data["edge_points"], dtype=np.float64)
    edge_weights = np.asarray(graph_data["edge_weights"], dtype=np.float64)

    for edge_id in range(len(edge_weights)):
        start = int(offsets[edge_id])
        end = int(offsets[edge_id + 1])
        points_yx = edge_points[start:end]
        graph.add_edge(
            int(edges_u[edge_id]),
            int(edges_v[edge_id]),
            pts=points_yx,
            weight=float(edge_weights[edge_id]),
        )

    return graph


class PyVoronoi:
    """High-level Voronoi diagram builder wrapping the native C++ library."""

    def __init__(self, scaling_factor: Optional[int] = None) -> None:
        log_action("Creating a VoronoiDiagram instance")
        self._diagram = _VoronoiDiagram()
        self._constructed = False
        self.scaling_factor = scaling_factor if scaling_factor is not None else 1

    @property
    def SCALING_FACTOR(self) -> int:
        return self.scaling_factor

    @SCALING_FACTOR.setter
    def SCALING_FACTOR(self, value: int) -> None:
        self.scaling_factor = value

    @property
    def constructed(self) -> int:
        return 1 if self._constructed else 0

    @constructed.setter
    def constructed(self, value: int) -> None:
        self._constructed = value == 1

    def add_point(self, point: Sequence[float]) -> None:
        if self._constructed:
            raise VoronoiException("Construct() has been called, can't add more elements")
        self._diagram.add_point(self._to_voronoi_point(point))

    def add_segment(self, segment: Sequence[Sequence[float]]) -> None:
        if self._constructed:
            raise VoronoiException("Construct() has been called, can't add more elements")
        self._diagram.add_segment(self._to_voronoi_segment(segment))

    def add_edges(self, edges) -> None:
        if self._constructed:
            raise VoronoiException("Construct() has been called, can't add more elements")
        self._diagram.add_edges(edges, self.scaling_factor)

    def construct(self):
        if self._constructed:
            raise VoronoiException("Construct() has already been called")
        self._constructed = True
        self._diagram.construct()
        self._diagram.map_vertex_indexes()
        self._diagram.map_edge_indexes()
        self._diagram.map_cell_indexes()

    def construct_realtime(self):
        if self._constructed:
            raise VoronoiException("Construct() has already been called")
        self._constructed = True
        self._diagram.construct()

    def get_point(self, index: int) -> list[int]:
        return _point_to_array(self._diagram.get_point(index))

    def get_segment(self, index: int) -> list[list[int]]:
        return _segment_to_array(self._diagram.get_segment(index))

    def get_vertex(self, index: int) -> Vertex:
        if index < 0 or index >= self.count_vertices():
            raise IndexError(index)
        c_vertex = self._diagram.get_vertex(index)
        return Vertex(c_vertex.X / self.scaling_factor, c_vertex.Y / self.scaling_factor)

    def get_edge(self, index: int) -> Edge:
        if index < 0 or index >= self.count_edges():
            raise IndexError(index)
        c_edge = self._diagram.get_edge(index)
        edge = Edge(c_edge.start, c_edge.end, c_edge.cell, c_edge.twin)
        edge.is_primary = c_edge.isPrimary is True
        edge.is_linear = c_edge.isLinear is True
        return edge

    def get_cell(self, index: int) -> Cell:
        if index < 0 or index >= self.count_cells():
            raise IndexError(index)
        c_cell = self._diagram.get_cell(index)
        cell = Cell(c_cell.cell_identifier, c_cell.site, c_cell.vertices, c_cell.edges, c_cell.source_category)
        cell.contains_point = c_cell.contains_point is True
        cell.contains_segment = c_cell.contains_segment is True
        cell.is_degenerate = c_cell.is_degenerate is True
        cell.is_open = c_cell.is_open is True
        return cell

    def count_points(self):
        return self._diagram.count_points()

    def count_segments(self):
        return self._diagram.count_segments()

    def count_vertices(self):
        return self._diagram.count_vertices()

    def count_edges(self):
        return self._diagram.count_edges()

    def count_cells(self):
        return self._diagram.count_cells()

    def get_points(self):
        for point in self._diagram.get_points():
            yield _point_to_array(point)

    def get_segments(self):
        for segment in self._diagram.get_segments():
            yield _segment_to_array(segment)

    def get_intersecting_segments(self):
        return self._diagram.get_intersecting_segments()

    def get_degenerate_segments(self):
        return self._diagram.get_degenerate_segments()

    def get_points_on_segments(self):
        return self._diagram.get_points_on_segments()

    def get_internal_ridges(self, exterior):
        exterior_array = np.asarray(exterior, dtype=np.float64)
        return self._diagram.get_internal_ridges(exterior_array, self.scaling_factor)

    def generate_internal_segments(self, exterior):
        return generate_internal_segments(exterior, self.scaling_factor)

    def generate_internal_graph(self, exterior):
        return generate_internal_graph(exterior, self.scaling_factor)

    @staticmethod
    def compute_internal_ridges(exterior, scaling_factor: int = 1):
        return generate_internal_segments(exterior, scaling_factor)

    def get_vertices(self):
        return [self.get_vertex(index) for index in range(self.count_vertices())]

    def enumerate_vertices(self):
        for index in range(self.count_vertices()):
            yield index, self.get_vertex(index)

    def get_edges(self):
        return [self.get_edge(index) for index in range(self.count_edges())]

    def enumerate_edges(self):
        for index in range(self.count_edges()):
            yield index, self.get_edge(index)

    def get_cells(self):
        return [self.get_cell(index) for index in range(self.count_cells())]

    def enumerate_cells(self):
        for index in range(self.count_cells()):
            yield index, self.get_cell(index)

    def _to_voronoi_segment(self, py_segment: Sequence[Sequence[float]]):
        return Segment(self._to_voronoi_point(py_segment[0]), self._to_voronoi_point(py_segment[1]))

    def _to_voronoi_point(self, py_point: Sequence[float]):
        return Point(self._to_voronoi_int(py_point[0]), self._to_voronoi_int(py_point[1]))

    def _to_voronoi_int(self, value):
        return round(value * self.scaling_factor)
