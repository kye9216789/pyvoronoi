#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "voronoi.hpp"

namespace py = pybind11;

namespace {

struct GraphPoint {
    double x;
    double y;

    bool operator==(const GraphPoint& other) const {
        return x == other.x && y == other.y;
    }
};

struct GraphEdgeRef {
    int u;
    int v;
};

struct GraphPath {
    int node_u;
    int node_v;
    std::vector<int> vertices;
    double weight;
};

struct PackedGraphData {
    std::vector<std::array<double, 2>> nodes_yx;
    std::vector<long long> edges_u;
    std::vector<long long> edges_v;
    std::vector<long long> edge_offsets;
    std::vector<std::array<double, 2>> edge_points_yx;
    std::vector<double> edge_weights;
};

double polyline_length(const std::vector<int>& path_vertices, const std::vector<GraphPoint>& points_xy) {
    double total = 0.0;
    if (path_vertices.size() < 2) {
        return total;
    }
    for (std::size_t i = 1; i < path_vertices.size(); ++i) {
        const auto& p0 = points_xy[static_cast<std::size_t>(path_vertices[i - 1])];
        const auto& p1 = points_xy[static_cast<std::size_t>(path_vertices[i])];
        const double dx = p1.x - p0.x;
        const double dy = p1.y - p0.y;
        total += std::sqrt(dx * dx + dy * dy);
    }
    return total;
}

PackedGraphData build_internal_graph_data(const std::vector<std::array<double, 4>>& ridges) {
    PackedGraphData packed;

    if (ridges.empty()) {
        packed.edge_offsets.push_back(0);
        return packed;
    }

    std::map<std::pair<double, double>, int> point_to_vertex;
    std::vector<GraphPoint> vertices_xy;
    vertices_xy.reserve(ridges.size() * 2);

    auto get_vertex_id = [&](double x, double y) -> int {
        const auto key = std::make_pair(x, y);
        const auto it = point_to_vertex.find(key);
        if (it != point_to_vertex.end()) {
            return it->second;
        }
        const int id = static_cast<int>(vertices_xy.size());
        vertices_xy.push_back(GraphPoint{x, y});
        point_to_vertex.emplace(key, id);
        return id;
    };

    std::map<std::pair<int, int>, int> undirected_edge_seen;
    std::vector<GraphEdgeRef> edges;
    edges.reserve(ridges.size());

    for (const auto& ridge : ridges) {
        const int u = get_vertex_id(ridge[0], ridge[1]);
        const int v = get_vertex_id(ridge[2], ridge[3]);
        if (u == v) {
            continue;
        }

        const int a = (u < v) ? u : v;
        const int b = (u < v) ? v : u;
        const auto key = std::make_pair(a, b);
        if (undirected_edge_seen.find(key) != undirected_edge_seen.end()) {
            continue;
        }
        undirected_edge_seen.emplace(key, static_cast<int>(edges.size()));
        edges.push_back(GraphEdgeRef{u, v});
    }

    if (edges.empty()) {
        packed.edge_offsets.push_back(0);
        return packed;
    }

    std::vector<std::vector<int>> adjacency(vertices_xy.size());
    for (int edge_id = 0; edge_id < static_cast<int>(edges.size()); ++edge_id) {
        const auto& e = edges[static_cast<std::size_t>(edge_id)];
        adjacency[static_cast<std::size_t>(e.u)].push_back(edge_id);
        adjacency[static_cast<std::size_t>(e.v)].push_back(edge_id);
    }

    std::vector<char> is_node(vertices_xy.size(), 0);
    std::vector<int> vertex_to_node(vertices_xy.size(), -1);

    for (std::size_t v = 0; v < adjacency.size(); ++v) {
        if (adjacency[v].size() != 2U) {
            is_node[v] = 1;
            vertex_to_node[v] = static_cast<int>(packed.nodes_yx.size());
            const auto& p = vertices_xy[v];
            packed.nodes_yx.push_back(std::array<double, 2>{p.y, p.x});
        }
    }

    std::vector<char> edge_visited(edges.size(), 0);
    std::vector<GraphPath> graph_paths;
    graph_paths.reserve(edges.size());

    auto other_vertex = [&](int edge_id, int current_vertex) -> int {
        const auto& e = edges[static_cast<std::size_t>(edge_id)];
        return (e.u == current_vertex) ? e.v : e.u;
    };

    auto extend_from_node = [&](int start_vertex, int first_edge_id) {
        if (edge_visited[static_cast<std::size_t>(first_edge_id)] != 0) {
            return;
        }

        std::vector<int> path_vertices;
        path_vertices.reserve(16);
        path_vertices.push_back(start_vertex);

        int current_vertex = start_vertex;
        int current_edge = first_edge_id;

        for (;;) {
            edge_visited[static_cast<std::size_t>(current_edge)] = 1;
            const int next_vertex = other_vertex(current_edge, current_vertex);
            path_vertices.push_back(next_vertex);

            if (is_node[static_cast<std::size_t>(next_vertex)] != 0) {
                const int node_u = vertex_to_node[static_cast<std::size_t>(start_vertex)];
                const int node_v = vertex_to_node[static_cast<std::size_t>(next_vertex)];
                graph_paths.push_back(GraphPath{node_u, node_v, path_vertices, polyline_length(path_vertices, vertices_xy)});
                return;
            }

            int next_edge = -1;
            const auto& incident = adjacency[static_cast<std::size_t>(next_vertex)];
            for (int candidate : incident) {
                if (candidate != current_edge && edge_visited[static_cast<std::size_t>(candidate)] == 0) {
                    next_edge = candidate;
                    break;
                }
            }

            if (next_edge < 0) {
                const int synthetic_node = static_cast<int>(packed.nodes_yx.size());
                vertex_to_node[static_cast<std::size_t>(next_vertex)] = synthetic_node;
                is_node[static_cast<std::size_t>(next_vertex)] = 1;
                const auto& p = vertices_xy[static_cast<std::size_t>(next_vertex)];
                packed.nodes_yx.push_back(std::array<double, 2>{p.y, p.x});
                const int node_u = vertex_to_node[static_cast<std::size_t>(start_vertex)];
                graph_paths.push_back(GraphPath{node_u, synthetic_node, path_vertices, polyline_length(path_vertices, vertices_xy)});
                return;
            }

            current_vertex = next_vertex;
            current_edge = next_edge;
        }
    };

    for (std::size_t v = 0; v < adjacency.size(); ++v) {
        if (is_node[v] == 0) {
            continue;
        }
        for (int edge_id : adjacency[v]) {
            if (edge_visited[static_cast<std::size_t>(edge_id)] == 0) {
                extend_from_node(static_cast<int>(v), edge_id);
            }
        }
    }

    for (int edge_id = 0; edge_id < static_cast<int>(edges.size()); ++edge_id) {
        if (edge_visited[static_cast<std::size_t>(edge_id)] != 0) {
            continue;
        }

        const int start_vertex = edges[static_cast<std::size_t>(edge_id)].u;
        std::vector<int> path_vertices;
        path_vertices.reserve(16);
        path_vertices.push_back(start_vertex);

        int current_vertex = start_vertex;
        int current_edge = edge_id;

        for (;;) {
            edge_visited[static_cast<std::size_t>(current_edge)] = 1;
            const int next_vertex = other_vertex(current_edge, current_vertex);
            path_vertices.push_back(next_vertex);

            int next_edge = -1;
            const auto& incident = adjacency[static_cast<std::size_t>(next_vertex)];
            for (int candidate : incident) {
                if (candidate != current_edge && edge_visited[static_cast<std::size_t>(candidate)] == 0) {
                    next_edge = candidate;
                    break;
                }
            }

            if (next_edge < 0 || next_vertex == start_vertex) {
                break;
            }

            current_vertex = next_vertex;
            current_edge = next_edge;
        }

        int loop_node = vertex_to_node[static_cast<std::size_t>(start_vertex)];
        if (loop_node < 0) {
            loop_node = static_cast<int>(packed.nodes_yx.size());
            vertex_to_node[static_cast<std::size_t>(start_vertex)] = loop_node;
            is_node[static_cast<std::size_t>(start_vertex)] = 1;
            const auto& p = vertices_xy[static_cast<std::size_t>(start_vertex)];
            packed.nodes_yx.push_back(std::array<double, 2>{p.y, p.x});
        }

        graph_paths.push_back(GraphPath{loop_node, loop_node, path_vertices, polyline_length(path_vertices, vertices_xy)});
    }

    packed.edge_offsets.reserve(graph_paths.size() + 1);
    packed.edge_offsets.push_back(0);

    for (const auto& path : graph_paths) {
        packed.edges_u.push_back(static_cast<long long>(path.node_u));
        packed.edges_v.push_back(static_cast<long long>(path.node_v));
        packed.edge_weights.push_back(path.weight);

        for (int vertex_id : path.vertices) {
            const auto& p = vertices_xy[static_cast<std::size_t>(vertex_id)];
            packed.edge_points_yx.push_back(std::array<double, 2>{p.y, p.x});
        }

        packed.edge_offsets.push_back(static_cast<long long>(packed.edge_points_yx.size()));
    }

    return packed;
}

py::dict pack_graph_data_to_dict(const PackedGraphData& packed) {
    py::array_t<double> nodes(py::array::ShapeContainer{
        static_cast<py::ssize_t>(packed.nodes_yx.size()),
        static_cast<py::ssize_t>(2),
    });
    auto nodes_out = nodes.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(packed.nodes_yx.size()); ++i) {
        nodes_out(i, 0) = packed.nodes_yx[static_cast<std::size_t>(i)][0];
        nodes_out(i, 1) = packed.nodes_yx[static_cast<std::size_t>(i)][1];
    }

    py::array_t<long long> edges_u(static_cast<py::ssize_t>(packed.edges_u.size()));
    py::array_t<long long> edges_v(static_cast<py::ssize_t>(packed.edges_v.size()));
    py::array_t<long long> edge_offsets(static_cast<py::ssize_t>(packed.edge_offsets.size()));
    py::array_t<double> edge_weights(static_cast<py::ssize_t>(packed.edge_weights.size()));
    py::array_t<double> edge_points(py::array::ShapeContainer{
        static_cast<py::ssize_t>(packed.edge_points_yx.size()),
        static_cast<py::ssize_t>(2),
    });

    {
        auto out = edges_u.mutable_unchecked<1>();
        for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(packed.edges_u.size()); ++i) {
            out(i) = packed.edges_u[static_cast<std::size_t>(i)];
        }
    }
    {
        auto out = edges_v.mutable_unchecked<1>();
        for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(packed.edges_v.size()); ++i) {
            out(i) = packed.edges_v[static_cast<std::size_t>(i)];
        }
    }
    {
        auto out = edge_offsets.mutable_unchecked<1>();
        for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(packed.edge_offsets.size()); ++i) {
            out(i) = packed.edge_offsets[static_cast<std::size_t>(i)];
        }
    }
    {
        auto out = edge_weights.mutable_unchecked<1>();
        for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(packed.edge_weights.size()); ++i) {
            out(i) = packed.edge_weights[static_cast<std::size_t>(i)];
        }
    }
    {
        auto out = edge_points.mutable_unchecked<2>();
        for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(packed.edge_points_yx.size()); ++i) {
            out(i, 0) = packed.edge_points_yx[static_cast<std::size_t>(i)][0];
            out(i, 1) = packed.edge_points_yx[static_cast<std::size_t>(i)][1];
        }
    }

    py::dict result;
    result["nodes"] = std::move(nodes);
    result["edges_u"] = std::move(edges_u);
    result["edges_v"] = std::move(edges_v);
    result["edge_offsets"] = std::move(edge_offsets);
    result["edge_points"] = std::move(edge_points);
    result["edge_weights"] = std::move(edge_weights);
    return result;
}

}

template <typename T>
void add_edges_from_rows(VoronoiDiagram& diagram, const py::buffer_info& info, int scaling_factor) {
    const auto* base = static_cast<const char*>(info.ptr);
    const auto row_stride = info.strides[0];
    const auto col_stride = info.strides[1];

    for (py::ssize_t row = 0; row < info.shape[0]; row += 2) {
        const auto* p0x_ptr = reinterpret_cast<const T*>(base + row * row_stride + 0 * col_stride);
        const auto* p0y_ptr = reinterpret_cast<const T*>(base + row * row_stride + 1 * col_stride);
        const auto* p1x_ptr = reinterpret_cast<const T*>(base + (row + 1) * row_stride + 0 * col_stride);
        const auto* p1y_ptr = reinterpret_cast<const T*>(base + (row + 1) * row_stride + 1 * col_stride);

        const auto p0x = static_cast<double>(*p0x_ptr);
        const auto p0y = static_cast<double>(*p0y_ptr);
        const auto p1x = static_cast<double>(*p1x_ptr);
        const auto p1y = static_cast<double>(*p1y_ptr);

        Point p0(static_cast<int>(std::llround(p0x * scaling_factor)), static_cast<int>(std::llround(p0y * scaling_factor)));
        Point p1(static_cast<int>(std::llround(p1x * scaling_factor)), static_cast<int>(std::llround(p1y * scaling_factor)));
        diagram.AddSegment(Segment(p0, p1));
    }
}

void add_edges_numpy(VoronoiDiagram& diagram, const py::array& edges, int scaling_factor) {
    py::buffer_info info = edges.request();

    if (info.ndim != 2 || info.shape[1] != 2) {
        throw std::invalid_argument("edges must be a NumPy array with shape (n, 2)");
    }
    if (info.shape[0] == 0) {
        return;
    }
    if (info.shape[0] % 2 != 0) {
        throw std::invalid_argument("edges array must contain an even number of rows; every pair of rows defines one edge");
    }

    const auto& fmt = info.format;
    if (fmt == py::format_descriptor<double>::format()) {
        add_edges_from_rows<double>(diagram, info, scaling_factor);
        return;
    }
    if (fmt == py::format_descriptor<float>::format()) {
        add_edges_from_rows<float>(diagram, info, scaling_factor);
        return;
    }
    if (fmt == py::format_descriptor<int>::format()) {
        add_edges_from_rows<int>(diagram, info, scaling_factor);
        return;
    }
    if (fmt == py::format_descriptor<long long>::format()) {
        add_edges_from_rows<long long>(diagram, info, scaling_factor);
        return;
    }

    throw std::invalid_argument("unsupported dtype for edges array; expected float32/float64/int32/int64");
}

std::vector<std::pair<double, double>> parse_polygon_numpy(const py::array& exterior) {
    py::buffer_info info = exterior.request();
    if (info.ndim != 2 || info.shape[1] != 2) {
        throw std::invalid_argument("exterior must be a NumPy array with shape (n, 2)");
    }

    if (info.shape[0] < 3) {
        return std::vector<std::pair<double, double>>{};
    }

    const auto* base = static_cast<const char*>(info.ptr);
    const auto row_stride = info.strides[0];
    const auto col_stride = info.strides[1];
    const auto& fmt = info.format;

    auto read_value = [&](py::ssize_t row, py::ssize_t col) -> double {
        const auto* p = base + row * row_stride + col * col_stride;
        if (fmt == py::format_descriptor<double>::format()) {
            return static_cast<double>(*reinterpret_cast<const double*>(p));
        }
        if (fmt == py::format_descriptor<float>::format()) {
            return static_cast<double>(*reinterpret_cast<const float*>(p));
        }
        if (fmt == py::format_descriptor<int>::format()) {
            return static_cast<double>(*reinterpret_cast<const int*>(p));
        }
        if (fmt == py::format_descriptor<long long>::format()) {
            return static_cast<double>(*reinterpret_cast<const long long*>(p));
        }
        throw std::invalid_argument("unsupported dtype for exterior array; expected float32/float64/int32/int64");
    };

    std::vector<std::pair<double, double>> polygon;
    polygon.reserve(static_cast<std::size_t>(info.shape[0]));
    for (py::ssize_t i = 0; i < info.shape[0]; ++i) {
        polygon.emplace_back(read_value(i, 0), read_value(i, 1));
    }

    if (polygon.size() > 1) {
        const auto& first = polygon.front();
        const auto& last = polygon.back();
        if (first.first == last.first && first.second == last.second) {
            polygon.pop_back();
        }
    }

    if (polygon.size() < 3) {
        return std::vector<std::pair<double, double>>{};
    }

    return polygon;
}

py::array_t<double> to_ridges_array(const std::vector<std::array<double, 4>>& kept) {
    py::array_t<double> output(py::array::ShapeContainer{
        static_cast<py::ssize_t>(kept.size()),
        static_cast<py::ssize_t>(2),
        static_cast<py::ssize_t>(2),
    });

    auto out = output.mutable_unchecked<3>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(kept.size()); ++i) {
        out(i, 0, 0) = kept[static_cast<std::size_t>(i)][0];
        out(i, 0, 1) = kept[static_cast<std::size_t>(i)][1];
        out(i, 1, 0) = kept[static_cast<std::size_t>(i)][2];
        out(i, 1, 1) = kept[static_cast<std::size_t>(i)][3];
    }

    return output;
}

void add_polygon_segments(VoronoiDiagram& diagram, const std::vector<std::pair<double, double>>& polygon, int scaling_factor) {
    const auto n = polygon.size();
    for (std::size_t i = 0; i < n; ++i) {
        const auto& p0 = polygon[i];
        const auto& p1 = polygon[(i + 1) % n];
        Point a(
            static_cast<int>(std::llround(p0.first * scaling_factor)),
            static_cast<int>(std::llround(p0.second * scaling_factor)));
        Point b(
            static_cast<int>(std::llround(p1.first * scaling_factor)),
            static_cast<int>(std::llround(p1.second * scaling_factor)));
        diagram.AddSegment(Segment(a, b));
    }
}

py::array_t<double> get_internal_ridges_numpy(VoronoiDiagram& diagram, const py::array& exterior, int scaling_factor) {
    if (scaling_factor <= 0) {
        throw std::invalid_argument("scaling_factor must be greater than 0");
    }

    const auto polygon = parse_polygon_numpy(exterior);
    if (polygon.size() < 3) {
        return py::array_t<double>(py::array::ShapeContainer{0, 2, 2});
    }

    return to_ridges_array(diagram.GetInternalRidgesNoMap(polygon, scaling_factor));
}

py::array_t<double> compute_internal_ridges_numpy(const py::array& exterior, int scaling_factor) {
    if (scaling_factor <= 0) {
        throw std::invalid_argument("scaling_factor must be greater than 0");
    }

    const auto polygon = parse_polygon_numpy(exterior);
    if (polygon.size() < 3) {
        return py::array_t<double>(py::array::ShapeContainer{0, 2, 2});
    }

    VoronoiDiagram diagram;
    add_polygon_segments(diagram, polygon, scaling_factor);
    diagram.Construct();

    return to_ridges_array(diagram.GetInternalRidgesNoMap(polygon, scaling_factor));
}

py::dict compute_internal_graph_numpy(const py::array& exterior, int scaling_factor) {
    if (scaling_factor <= 0) {
        throw std::invalid_argument("scaling_factor must be greater than 0");
    }

    const auto polygon = parse_polygon_numpy(exterior);
    if (polygon.size() < 3) {
        PackedGraphData empty;
        empty.edge_offsets.push_back(0);
        return pack_graph_data_to_dict(empty);
    }

    VoronoiDiagram diagram;
    add_polygon_segments(diagram, polygon, scaling_factor);
    diagram.Construct();

    const auto ridges = diagram.GetInternalRidgesNoMap(polygon, scaling_factor);
    return pack_graph_data_to_dict(build_internal_graph_data(ridges));
}

PYBIND11_MODULE(_pyvoronoi, m) {
    m.doc() = "pybind11 bindings for pyvoronoi";

    py::class_<Point>(m, "Point")
        .def(py::init<int, int>(), py::arg("x") = 0, py::arg("y") = 0)
        .def_readwrite("X", &Point::X)
        .def_readwrite("Y", &Point::Y);

    py::class_<Segment>(m, "Segment")
        .def(py::init<Point, Point>(), py::arg("a") = Point(), py::arg("b") = Point())
        .def_readwrite("p0", &Segment::p0)
        .def_readwrite("p1", &Segment::p1);

    py::class_<c_Vertex>(m, "c_Vertex")
        .def(py::init<double, double>(), py::arg("x") = 0.0, py::arg("y") = 0.0)
        .def_readwrite("X", &c_Vertex::X)
        .def_readwrite("Y", &c_Vertex::Y);

    py::class_<c_Edge>(m, "c_Edge")
        .def(
            py::init<long long, long long, bool, bool, long long, long long>(),
            py::arg("start") = -1,
            py::arg("end") = -1,
            py::arg("is_primary") = false,
            py::arg("is_linear") = false,
            py::arg("cell") = -1,
            py::arg("twin") = -1)
        .def_readwrite("start", &c_Edge::start)
        .def_readwrite("end", &c_Edge::end)
        .def_readwrite("isPrimary", &c_Edge::isPrimary)
        .def_readwrite("isLinear", &c_Edge::isLinear)
        .def_readwrite("cell", &c_Edge::cell)
        .def_readwrite("twin", &c_Edge::twin);

    py::class_<c_Cell>(m, "c_Cell")
        .def(
            py::init<size_t, size_t, bool, bool, bool, int>(),
            py::arg("cell_identifier") = static_cast<size_t>(-1),
            py::arg("site") = static_cast<size_t>(-1),
            py::arg("contains_point") = false,
            py::arg("contains_segment") = false,
            py::arg("is_open") = false,
            py::arg("source_category") = -1)
        .def_readwrite("cell_identifier", &c_Cell::cell_identifier)
        .def_readwrite("site", &c_Cell::site)
        .def_readwrite("contains_point", &c_Cell::contains_point)
        .def_readwrite("contains_segment", &c_Cell::contains_segment)
        .def_readwrite("is_open", &c_Cell::is_open)
        .def_readwrite("is_degenerate", &c_Cell::is_degenerate)
        .def_readwrite("vertices", &c_Cell::vertices)
        .def_readwrite("edges", &c_Cell::edges)
        .def_readwrite("source_category", &c_Cell::source_category);

    py::class_<VoronoiDiagram>(m, "VoronoiDiagram")
        .def(py::init<>())
        .def("add_point", &VoronoiDiagram::AddPoint)
        .def("add_segment", &VoronoiDiagram::AddSegment)
        .def("add_edges", &add_edges_numpy, py::arg("edges"), py::arg("scaling_factor"))
        .def("construct", &VoronoiDiagram::Construct)
        .def("get_points", &VoronoiDiagram::GetPoints)
        .def("get_segments", &VoronoiDiagram::GetSegments)
        .def("get_intersecting_segments", &VoronoiDiagram::GetIntersectingSegments)
        .def("get_degenerate_segments", &VoronoiDiagram::GetDegenerateSegments)
        .def("get_points_on_segments", &VoronoiDiagram::GetPointsOnSegments)
        .def("map_vertex_indexes", &VoronoiDiagram::MapVertexIndexes)
        .def("map_edge_indexes", &VoronoiDiagram::MapEdgeIndexes)
        .def("map_cell_indexes", &VoronoiDiagram::MapCellIndexes)
        .def("count_points", &VoronoiDiagram::CountPoints)
        .def("count_segments", &VoronoiDiagram::CountSegments)
        .def("count_vertices", &VoronoiDiagram::CountVertices)
        .def("count_edges", &VoronoiDiagram::CountEdges)
        .def("count_cells", &VoronoiDiagram::CountCells)
        .def("get_point", &VoronoiDiagram::GetPoint)
        .def("get_segment", &VoronoiDiagram::GetSegment)
        .def("get_vertex", &VoronoiDiagram::GetVertex)
        .def("get_edge", &VoronoiDiagram::GetEdge)
        .def("get_cell", &VoronoiDiagram::GetCell)
        .def("get_internal_ridges", &get_internal_ridges_numpy, py::arg("exterior"), py::arg("scaling_factor"))
        .def("reset", &VoronoiDiagram::Reset);

    m.def("compute_internal_ridges", &compute_internal_ridges_numpy, py::arg("exterior"), py::arg("scaling_factor"));
    m.def("generate_internal_segments", &compute_internal_ridges_numpy, py::arg("exterior"), py::arg("scaling_factor"));
    m.def("generate_internal_graph", &compute_internal_graph_numpy, py::arg("exterior"), py::arg("scaling_factor"));
}
