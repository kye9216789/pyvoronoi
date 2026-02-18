#!/usr/bin/python
"""
Tests for PyVoronoi wrapper library.
"""

from __future__ import print_function
from unittest import TestCase, main

import pyvoronoi
import numpy as np

class TestPyVoronoiModule(TestCase):
    def test_has_classes(self):
        self.assertTrue(hasattr(pyvoronoi, 'PyVoronoi'))

    def test_scaling_factor(self):
        pv = pyvoronoi.PyVoronoi()
        self.assertTrue(pv.SCALING_FACTOR == 1)

        pv = pyvoronoi.PyVoronoi(10)
        self.assertTrue(pv.SCALING_FACTOR == 10)

        pv = pyvoronoi.PyVoronoi(1)
        self.assertTrue(pv.SCALING_FACTOR == 1)

class TestPyVoronoiAdd(TestCase):
    def test_add_point(self):
        factor = 10
        inputPoint = [0.5, 1]
        pv = pyvoronoi.PyVoronoi(factor)
        pv.add_point(inputPoint)
        points = list(pv.get_points())
        self.assertTrue(len(points) == 1)
        self.assertTrue(points[0][0] == inputPoint[0] * factor)
        self.assertTrue(points[0][1] == inputPoint[1] * factor)

    def test_add_segment(self):
        factor = 10
        segment = [[0.5, 1], [0, 2]]
        pv = pyvoronoi.PyVoronoi(factor)
        pv.add_segment(segment)
        segments = list(pv.get_segments())
        self.assertTrue(len(segments) == 1)
        self.assertTrue(segments[0] == [
            [segment[0][0] * factor, segment[0][1] * factor],
            [segment[1][0] * factor, segment[1][1] * factor],
        ])

    def test_add_edges_numpy(self):
        factor = 10
        edges = np.array(
            [
                [0.5, 1.0],
                [0.0, 2.0],
                [1.2, 0.2],
                [3.0, 4.0],
            ],
            dtype=np.float64,
        )
        pv = pyvoronoi.PyVoronoi(factor)
        pv.add_edges(edges)
        segments = list(pv.get_segments())
        self.assertEqual(2, len(segments))
        self.assertEqual([[5, 10], [0, 20]], segments[0])
        self.assertEqual([[12, 2], [30, 40]], segments[1])

    def test_add_point_after_construct(self):
        pv = pyvoronoi.PyVoronoi()
        pv.construct()
        self.assertRaises(pyvoronoi.VoronoiException, pv.add_point, [0, 0])

    def test_add_segment_rounding(self):
        factor = 10
        segment = [[0.59, 1.09], [0, 2]]
        pv = pyvoronoi.PyVoronoi(factor)
        pv.add_segment(segment)
        segments = list(pv.get_segments())
        self.assertTrue(len(segments) == 1)
        self.assertTrue(segments[0] == [
            [round(segment[0][0] * factor), round(segment[0][1] * factor)],
            [segment[1][0] * factor, segment[1][1] * factor],
        ])

class TestPyVoronoiconstruct(TestCase):
    def test_square(self):
        pv = pyvoronoi.PyVoronoi(1)
        pv.add_segment([[0, 0], [0, 1]])
        pv.add_segment([[0, 1], [1, 1]])
        pv.add_segment([[1, 1], [1, 0]])
        pv.add_segment([[1, 0], [0, 0]])
        pv.construct()
        edges = pv.get_edges()
        vertices = pv.get_vertices()
        cells = pv.get_cells()
        self.assertTrue(len(cells) == 8)
        self.assertTrue(len([i for i in edges if i.is_primary == True]) == 8)
        self.assertTrue(len(vertices) == 5)
        self.assertTrue(len(cells[0].edges) == 2)

    def test_rectangle(self):
        pv = pyvoronoi.PyVoronoi(1)
        pv.add_segment([[0, 0], [0, 2]])
        pv.add_segment([[0, 2], [1, 2]])
        pv.add_segment([[1, 2], [1, 0]])
        pv.add_segment([[1, 0], [0, 0]])
        pv.construct()
        edges = pv.get_edges()
        vertices = pv.get_vertices()
        cells = pv.get_cells()
        self.assertTrue(len(cells) == 8)
        self.assertTrue(len([i for i in edges if i.is_primary == True]) == 10)
        self.assertTrue(len(vertices) == 6)
        self.assertTrue(len(list(filter(lambda e: edges[e].is_primary, cells[1].edges))) == 3)
        self.assertTrue(len(list(filter(lambda e: edges[e].is_primary, cells[3].edges))) == 2)

    def test_twins(self):
        """
        Validate that the twin attribute is consistent.
        :return:
        """
        pv = pyvoronoi.PyVoronoi(1)
        pv.add_point([5,5])
        pv.add_segment([[0,0],[0,10]])
        pv.add_segment([[0,0],[10,0]])
        pv.add_segment([[0,10],[10,10]])
        pv.add_segment([[10,0],[10,10]])
        pv.construct()
        edges = pv.get_edges()
        for i in range(len(edges)):
            edge = edges[i]
            self.assertTrue(edges[edge.twin].twin == i)
        cells = pv.get_cells()

    def test_cells_vertices_duplication(self):
        """
        Validate that the first and last vertex are the same on cells.
        :return:
        """
        pv = pyvoronoi.PyVoronoi(1)
        pv.add_point([5,5])
        pv.add_segment([[0,0],[0,10]])
        pv.add_segment([[0,0],[10,0]])
        pv.add_segment([[0,10],[10,10]])
        pv.add_segment([[10,0],[10,10]])
        pv.construct()
        cells = pv.get_cells()
        vertices = pv.get_vertices()
        cell = cells[5]
        self.assertNotEqual(cell.vertices[-2], cell.vertices[-1])
        self.assertEqual(cell.vertices[0], cell.vertices[-1])

    def test_output_enumeration(self):
        """
        Validate that the first and last vertex are the same on cells.
        :return:
        """
        pv = pyvoronoi.PyVoronoi(1)
        pv.add_point([5,5])
        pv.add_segment([[0,0],[0,10]])
        pv.add_segment([[0,0],[10,0]])
        pv.add_segment([[0,10],[10,10]])
        pv.add_segment([[10,0],[10,10]])
        pv.construct()
        cells = pv.get_cells()
        edges = pv.get_edges()
        vertices = pv.get_vertices()

        # Check that enumerators return the same content that list
        for index, vertex in pv.enumerate_vertices():
            self.assertEqual(vertex.X, vertices[index].X)
            self.assertEqual(vertex.Y, vertices[index].Y)

        for index, edge in pv.enumerate_edges():
            self.assertEqual(edge.start, edges[index].start)
            self.assertEqual(edge.end, edges[index].end)

        for index, cell in pv.enumerate_cells():
            self.assertEqual(cell.cell_identifier, cells[index].cell_identifier)
            self.assertEqual(cell.site, cells[index].site)


    def test_vertex_reference_for_edges(self):
        """
        Test the node edge have both ends not referencing a vertex.
        :return:
        """
        pv = pyvoronoi.PyVoronoi(1)
        pv.add_point([5,5])
        pv.add_segment([[0,0],[0,10]])
        pv.add_segment([[0,0],[10,0]])
        pv.add_segment([[0,10],[10,10]])
        pv.add_segment([[10,0],[10,10]])
        pv.construct()
        edges = pv.get_edges()
        cells = pv.get_cells()
        for i in range(len(edges)):
            edge = edges[i]
            if cells[edge.cell].is_open == False:
                self.assertTrue(edge.start != -1 and edge.end != -1)

    def test_edge_vertices_indexes(self):
        """
        Test the node edge have both ends not referencing a vertex.
        :return:
        """
        pv = pyvoronoi.PyVoronoi(1)
        pv.add_segment([[0,0],[0,10]])
        pv.add_segment([[0,0],[10,0]])
        pv.add_segment([[0,10],[10,10]])
        pv.add_segment([[10,0],[10,10]])
        pv.construct()
        edges = pv.get_edges()

        vertices_count = 0

        for i in range(len(edges)):
            edge = edges[i]
            vertices = [edge.start, edge.end]
            for v in vertices:
                if v == -1:
                    vertices_count += 1

        self.assertTrue(vertices_count == 16)

    def test_vertex_reference_for_cells(self):
        """
        Test that cells reference at least one vertex.
        :return:
        """
        pv = pyvoronoi.PyVoronoi(1)
        pv.add_point([5,5])
        pv.add_segment([[0,0],[0,10]])
        pv.add_segment([[0,0],[10,0]])
        pv.add_segment([[0,10],[10,10]])
        pv.add_segment([[10,0],[10,10]])
        pv.construct()
        cells = pv.get_cells()
        for i in range(len(cells)):
            cell = cells[i]
            if not cell.is_degenerate:
                valid_vertices = [v for v in cell.vertices if v != -1]
                self.assertTrue(len(valid_vertices) > 0)

    def test_retrieve_input(self):
        pv = pyvoronoi.PyVoronoi(1)

        p1 = [5, 5]
        p2 = [0, 0]
        s1 = [[10, 10], [20, 20]]
        s2 = [[10, 10], [20, 20]]
        s3 = [[-10, -10], [-20, -20]]

        pv.add_point(p1)
        pv.add_point(p2)
        pv.add_segment(s1)
        pv.add_segment(s2)
        pv.add_segment(s3)


        pv.construct()

        self.assertTrue(2 == len(list(pv.get_points())))
        self.assertEqual(2, pv.count_points())
        self.assertTrue(3 == len(list(pv.get_segments())))
        self.assertEqual(3, pv.count_segments())
        self.assertEqual(p1, pv.get_point(0))
        self.assertEqual(p2, pv.get_point(1))
        self.assertEqual(s1, pv.get_segment(0))
        self.assertEqual(s2, pv.get_segment(1))

    def test_get_raises_indexerror(self):
        pv = pyvoronoi.PyVoronoi(1)
        with self.assertRaises(IndexError):
            pv.get_edge(0) # shouldn't crash
        with self.assertRaises(IndexError):
            pv.get_vertex(0) # shouldn't crash
        with self.assertRaises(IndexError):
            pv.get_cell(0) # shouldn't crash

    def test_objects_dont_share_data(self):
        pv = pyvoronoi.PyVoronoi(1)
        pv.add_point([5, 5])
        pv.add_segment([[0, 0], [0, 10]])

        pv2 = pyvoronoi.PyVoronoi(1)
        pv2.add_point([9, 9])
        pv2.add_segment([[1, 1], [1, 9]])

        pv.construct()
        pv2.construct()

        self.assertEqual([[5, 5]], list(pv.get_points()))
        self.assertEqual([[[0, 0], [0, 10]]], list(pv.get_segments()))
        self.assertEqual([[9, 9]], list(pv2.get_points()))
        self.assertEqual([[[1, 1], [1, 9]]], list(pv2.get_segments()))


class TestInputSegmentIntersects(TestCase):
    def test_true_intersection_1(self):
        """
         Test that our 2 intersecting segments are detected among two other segment that do not intersect anything.
         :return:
        """

        pv = pyvoronoi.PyVoronoi(1)

        # Those first two segments do not intersect
        pv.add_segment([[-6, -6], [-10, -10]])
        pv.add_segment([[6, 6], [10, 10]])

        # Those two segments intersect but do not intersect the first two segments
        pv.add_segment([[0, 0], [10, 0]])
        pv.add_segment([[5, -5], [5, 10]])

        # The output should be our two intersecting segments indexed, without duplicating any idenfiers
        intersecting_segments = pv.get_intersecting_segments()
        self.assertEqual(2, len(intersecting_segments))
        self.assertEqual(2, intersecting_segments[0])
        self.assertEqual(3, intersecting_segments[1])

    def test_true_intersection_2(self):
        """
         Test that our 2 intersecting segments are detected
         :return:
        """

        pv = pyvoronoi.PyVoronoi(1)

        pv.add_segment([[10, 10], [-10, -10]])
        pv.add_segment([[-10, 10], [10, -10]])

        intersecting_segments = pv.get_intersecting_segments()
        self.assertEqual(2, len(intersecting_segments))
        self.assertEqual(0, intersecting_segments[0])
        self.assertEqual(1, intersecting_segments[1])

    def test_end_intersection_3(self):
        """
         Test that our 2 intersecting segments are detected
         :return:
        """

        pv = pyvoronoi.PyVoronoi(1)

        pv.add_segment([[0, 0], [10, 0]])
        pv.add_segment([[5, 0], [5, 10]])

        intersecting_segments = pv.get_intersecting_segments()
        self.assertEqual(2, len(intersecting_segments))


    def test_true_intersection_at_ends_is_disregarded_horizontal(self):
        """
        Test that the intersection test returns false when line intersects at endpoints.
        In that case, they touch, but do not intersect.
        :return:
        """
        pv = pyvoronoi.PyVoronoi(1)

        # Those first two segments not intersect
        pv.add_segment([[0, 0], [10, 0]])
        pv.add_segment([[-10, 0], [0, 0]])

        intersecting_segments = pv.get_intersecting_segments()
        self.assertEqual(0, len(intersecting_segments))

    def test_true_intersection_at_ends_is_disregarded_vertical(self):
        """
        Test that the intersection test returns false when line intersects at endpoints.
        In that case, they touch, but do not intersect.
        :return:
        """
        pv = pyvoronoi.PyVoronoi(1)

        # Those first two segments not intersect
        pv.add_segment([[0, 0], [0, 10]])
        pv.add_segment([[0, -10], [0, 0]])

        intersecting_segments = pv.get_intersecting_segments()
        self.assertEqual(0, len(intersecting_segments))


    def test_true_intersection_at_ends_is_disregarded(self):
        """
        Test that the intersection test returns false when line intersects at endpoints.
        In that case, they touch, but do not intersect.
        :return:
        """
        pv = pyvoronoi.PyVoronoi(1)

        # Those first two segments not intersect
        pv.add_segment([[0, 0], [5, 5]])
        pv.add_segment([[5, 5], [10, 10]])

        intersecting_segments = pv.get_intersecting_segments()
        self.assertEqual(0, len(intersecting_segments))


    def test_colinearity_intersects_horizontal(self):
        pv = pyvoronoi.PyVoronoi(1)

        # Those two segments overlap on 0,0 --> 5,0
        pv.add_segment([[0, 0], [10, 0]])
        pv.add_segment([[-10, 0], [5, 0]])

        # Those two segments not intersect or overlap anything
        pv.add_segment([[-6, -6], [-10, -10]])
        pv.add_segment([[6, 6], [10, 10]])

        # Validate that the two segments that intersects on 0 --> 5 intersect
        intersecting_segments = pv.get_intersecting_segments()
        self.assertEqual(2, len(intersecting_segments))
        self.assertEqual(0, intersecting_segments[0])
        self.assertEqual(1, intersecting_segments[1])

    def test_colinearity_intersects_vertical(self):
        pv = pyvoronoi.PyVoronoi(1)

        # Those two segments overlap on 0,0 --> 5,0
        pv.add_segment([[0, 0], [0, 10]])
        pv.add_segment([[0, -10], [0, 5]])

        # Those two segments not intersect or overlap anything
        pv.add_segment([[-6, -6], [-10, -10]])
        pv.add_segment([[6, 6], [10, 10]])

        # Validate that the two segments that intersects on 0 --> 5 intersect
        intersecting_segments = pv.get_intersecting_segments()
        self.assertEqual(2, len(intersecting_segments))
        self.assertEqual(0, intersecting_segments[0])
        self.assertEqual(1, intersecting_segments[1])


    def test_colinearity_intersects(self):
        pv = pyvoronoi.PyVoronoi(1)

        # Those two segments not intersect or overlap anything
        pv.add_segment([[-6, -6], [-10, -10]])
        pv.add_segment([[6, 6], [10, 10]])

        # Those two segments overlap
        pv.add_segment([[0, 0], [2, 2]])
        pv.add_segment([[1, 1], [3, 3]])


        # Validate that the two segments that intersects on 0 --> 5 intersect
        intersecting_segments = pv.get_intersecting_segments()
        self.assertEqual(2, len(intersecting_segments))
        self.assertEqual(2, intersecting_segments[0])
        self.assertEqual(3, intersecting_segments[1])

class TestDegeneratedInputSegment(TestCase):
    def test_no_degenerate_segment(self):
        pv = pyvoronoi.PyVoronoi(1)

        # Those two segments not intersect or overlap anything
        pv.add_segment([[-6, -6], [-10, -10]])
        pv.add_segment([[6, 6], [10, 10]])

        invalid_segments = pv.get_degenerate_segments()
        self.assertEqual(0, len(invalid_segments))

    def test_degenerate_segment(self):
        pv = pyvoronoi.PyVoronoi(1)

        # Those two segments not intersect or overlap anything
        pv.add_segment([[-6, -6], [-10, -10]])
        pv.add_segment([[6, 6], [6, 6]])

        invalid_segments = pv.get_degenerate_segments()
        # s = pv.get_segment(invalid_segments[0])
        self.assertEqual(1, len(invalid_segments))


class TestInternalGraphSmoothing(TestCase):
    @staticmethod
    def _point_in_polygon_or_boundary(x: float, y: float, polygon_xy: np.ndarray) -> bool:
        eps = 1e-9
        inside = False
        n = len(polygon_xy)
        for i in range(n):
            j = (i - 1) % n
            ax, ay = polygon_xy[j]
            bx, by = polygon_xy[i]

            cross = abs((bx - ax) * (y - ay) - (by - ay) * (x - ax))
            if cross <= eps and min(ax, bx) - eps <= x <= max(ax, bx) + eps and min(ay, by) - eps <= y <= max(ay, by) + eps:
                return True

            cond = (ay > y) != (by > y)
            if cond:
                x_intersect = ax + (y - ay) * (bx - ax) / (by - ay)
                if x_intersect >= x:
                    inside = not inside
        return inside

    def test_graph_resampling_preserves_endpoints_and_polygon_bounds(self):
        exterior = np.asarray(np.load("cnt.npy", allow_pickle=True), dtype=np.float64)
        if np.allclose(exterior[0], exterior[-1]):
            exterior = exterior[:-1]

        graph = pyvoronoi.generate_internal_graph(
            exterior,
            scaling_factor=1000000,
            resample_spacing=2.0,
            smooth_iterations=2,
            enforce_within_polygon=True,
        )

        self.assertGreater(graph.number_of_edges(), 0)

        for u, v, attrs in graph.edges(data=True):
            pts = np.asarray(attrs["pts"], dtype=np.float64)
            self.assertGreaterEqual(len(pts), 2)
            self.assertTrue(np.allclose(pts[0], np.asarray(graph.nodes[u]["o"], dtype=np.float64)))
            self.assertTrue(np.allclose(pts[-1], np.asarray(graph.nodes[v]["o"], dtype=np.float64)))

            for point_xy in pts:
                x = float(point_xy[0])
                y = float(point_xy[1])
                self.assertTrue(self._point_in_polygon_or_boundary(x, y, exterior))

class TestInputPointOnInputSegment(TestCase):
    def test_no_point_on_segment(self):
        pv = pyvoronoi.PyVoronoi(1)

        # Those two segments not intersect or overlap anything
        pv.add_segment([[-6, -6], [-10, -10]])
        pv.add_segment([[6, 6], [10, 10]])
        pv.add_point([0,0])

        invalid_points = pv.get_points_on_segments()
        self.assertEqual(0, len(invalid_points))

    def test_point_on_segment_end_point(self):
        """
        PyVoronoi does not consider the point on the line if it is equal to an end point.
        :return:
        """
        pv = pyvoronoi.PyVoronoi(1)

        # Those two segments not intersect or overlap anything
        pv.add_segment([[-6, -6], [-10, -10]])
        pv.add_segment([[6, 6], [10, 10]])
        pv.add_point([10, 10])

        invalid_points = pv.get_points_on_segments()
        self.assertEqual(0, len(invalid_points))



    def test_point_on_segment_factor10(self):
        pv = pyvoronoi.PyVoronoi(10)

        # Those two segments not intersect or overlap anything
        pv.add_segment([[-6, -6], [-10, -10]])
        pv.add_segment([[6.6, 6.6], [10.1, 10.1]])
        pv.add_point([0, 0])
        pv.add_point([7.7, 7.7])

        invalid_points = pv.get_points_on_segments()
        self.assertEqual(1, len(invalid_points))
        self.assertEqual(1, invalid_points[0])

    def test_scenario(self):
        pv = pyvoronoi.PyVoronoi(1)
        # Those two segments not intersect or overlap anything
        pv.add_segment([[-8433001, 5672399], [-8418599, 5672399]])
        pv.add_segment([[-8433001, 5672399], [-8433001, 5687401]])
        # pv.add_segment([[-8418599, 5687401], [-8418599, 5672399]])



        invalid_segments = pv.get_intersecting_segments()
        self.assertEqual(0, len(invalid_segments))

def run_tests():
    main()

if __name__ == '__main__':
    run_tests()
