"""Data container classes for pyvoronoi."""


class VoronoiException(Exception):
    """Base exception for Voronoi-related errors."""

    pass


class Vertex:
    """Voronoi diagram vertex with x/y coordinates."""

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    @property
    def X(self) -> float:
        return self.x

    @X.setter
    def X(self, value: float) -> None:
        self.x = value

    @property
    def Y(self) -> float:
        return self.y

    @Y.setter
    def Y(self, value: float) -> None:
        self.y = value


class Edge:
    """Voronoi diagram edge connecting two vertices."""

    def __init__(self, start: int, end: int, cell: int, twin: int) -> None:
        self.start = start
        self.end = end
        self.cell = cell
        self.twin = twin
        self.is_primary = False
        self.is_linear = False


class Cell:
    """Voronoi diagram cell associated with an input site."""

    def __init__(
        self,
        cell_identifier: int,
        site: int,
        vertices: list[int],
        edges: list[int],
        source_category: int,
    ) -> None:
        self.cell_identifier = cell_identifier
        self.site = site
        self.source_category = source_category
        self.vertices = list(vertices)
        self.edges = list(edges)
        self.contains_point = False
        self.contains_segment = False
        self.is_open = False
        self.is_degenerate = False
        if self.vertices:
            self.vertices.append(self.vertices[0])
