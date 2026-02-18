#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <utility>
#include <vector>

namespace pyvoronoi_internal {

struct GraphPoint {
    double x;
    double y;
};

struct GraphEdgeRef {
    int u;
    int v;
};

struct GraphPath {
    int node_u;
    int node_v;
    std::vector<int> vertices;
};

struct GraphBuildOptions {
    double resample_spacing;
    int smooth_iterations;
    bool enforce_within_polygon;
};

struct PackedGraphData {
    std::vector<std::array<double, 2>> nodes_xy;
    std::vector<long long> edges_u;
    std::vector<long long> edges_v;
    std::vector<long long> edge_offsets;
    std::vector<std::array<double, 2>> edge_points_xy;
    std::vector<double> edge_weights;
};

inline double polyline_length(const std::vector<GraphPoint>& points_xy) {
    double total = 0.0;
    if (points_xy.size() < 2) {
        return total;
    }
    for (std::size_t i = 1; i < points_xy.size(); ++i) {
        const auto& p0 = points_xy[i - 1];
        const auto& p1 = points_xy[i];
        const double dx = p1.x - p0.x;
        const double dy = p1.y - p0.y;
        total += std::sqrt(dx * dx + dy * dy);
    }
    return total;
}

inline GraphPoint interpolate_point(const GraphPoint& p0, const GraphPoint& p1, double t) {
    return GraphPoint{p0.x + t * (p1.x - p0.x), p0.y + t * (p1.y - p0.y)};
}

inline double distance_squared(const GraphPoint& a, const GraphPoint& b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return dx * dx + dy * dy;
}

inline double cross_value(const GraphPoint& a, const GraphPoint& b, const GraphPoint& p) {
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

inline bool point_on_segment(const GraphPoint& p, const GraphPoint& a, const GraphPoint& b, double eps) {
    const double cross = std::abs(cross_value(a, b, p));
    if (cross > eps) {
        return false;
    }
    const double min_x = (a.x < b.x) ? a.x : b.x;
    const double max_x = (a.x > b.x) ? a.x : b.x;
    const double min_y = (a.y < b.y) ? a.y : b.y;
    const double max_y = (a.y > b.y) ? a.y : b.y;
    return p.x >= (min_x - eps) && p.x <= (max_x + eps) && p.y >= (min_y - eps) && p.y <= (max_y + eps);
}

inline bool point_in_polygon_or_boundary(const GraphPoint& p, const std::vector<std::pair<double, double>>& polygon_xy) {
    if (polygon_xy.size() < 3) {
        return false;
    }

    constexpr double eps = 1e-12;
    bool inside = false;
    const std::size_t n = polygon_xy.size();

    for (std::size_t i = 0, j = n - 1; i < n; j = i++) {
        const GraphPoint a{polygon_xy[j].first, polygon_xy[j].second};
        const GraphPoint b{polygon_xy[i].first, polygon_xy[i].second};

        if (point_on_segment(p, a, b, eps)) {
            return true;
        }

        const bool cond = ((a.y > p.y) != (b.y > p.y));
        if (cond) {
            const double x_intersect = a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y);
            if (x_intersect >= p.x) {
                inside = !inside;
            }
        }
    }

    return inside;
}

inline GraphPoint closest_point_on_polygon(const GraphPoint& p, const std::vector<std::pair<double, double>>& polygon_xy) {
    GraphPoint best = p;
    double best_dist2 = std::numeric_limits<double>::infinity();
    const std::size_t n = polygon_xy.size();

    for (std::size_t i = 0; i < n; ++i) {
        const GraphPoint a{polygon_xy[i].first, polygon_xy[i].second};
        const GraphPoint b{polygon_xy[(i + 1) % n].first, polygon_xy[(i + 1) % n].second};
        const double dx = b.x - a.x;
        const double dy = b.y - a.y;
        const double denom = dx * dx + dy * dy;

        double t = 0.0;
        if (denom > 0.0) {
            t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / denom;
            if (t < 0.0) {
                t = 0.0;
            } else if (t > 1.0) {
                t = 1.0;
            }
        }

        const GraphPoint candidate{a.x + t * dx, a.y + t * dy};
        const double d2 = distance_squared(p, candidate);
        if (d2 < best_dist2) {
            best_dist2 = d2;
            best = candidate;
        }
    }

    return best;
}

inline std::vector<GraphPoint> chaikin_smooth_open(const std::vector<GraphPoint>& points_xy, int iterations) {
    if (points_xy.size() < 3 || iterations <= 0) {
        return points_xy;
    }

    std::vector<GraphPoint> current = points_xy;
    for (int iter = 0; iter < iterations; ++iter) {
        if (current.size() < 3) {
            break;
        }

        std::vector<GraphPoint> next;
        next.reserve(current.size() * 2);
        next.push_back(current.front());

        for (std::size_t i = 0; i + 1 < current.size(); ++i) {
            const auto& p = current[i];
            const auto& q = current[i + 1];
            next.push_back(GraphPoint{0.75 * p.x + 0.25 * q.x, 0.75 * p.y + 0.25 * q.y});
            next.push_back(GraphPoint{0.25 * p.x + 0.75 * q.x, 0.25 * p.y + 0.75 * q.y});
        }

        next.push_back(current.back());
        current.swap(next);
    }

    return current;
}

inline std::vector<GraphPoint> resample_evenly(const std::vector<GraphPoint>& points_xy, double spacing) {
    if (points_xy.size() < 2) {
        return points_xy;
    }

    std::vector<double> cumulative(points_xy.size(), 0.0);
    for (std::size_t i = 1; i < points_xy.size(); ++i) {
        const auto& p0 = points_xy[i - 1];
        const auto& p1 = points_xy[i];
        const double dx = p1.x - p0.x;
        const double dy = p1.y - p0.y;
        cumulative[i] = cumulative[i - 1] + std::sqrt(dx * dx + dy * dy);
    }

    const double total = cumulative.back();
    if (total <= 0.0) {
        return std::vector<GraphPoint>{points_xy.front(), points_xy.back()};
    }

    long long interval_count = 1;
    if (spacing > 0.0) {
        interval_count = static_cast<long long>(std::ceil(total / spacing));
        if (interval_count < 1) {
            interval_count = 1;
        }
    } else {
        interval_count = static_cast<long long>(points_xy.size() - 1);
        if (interval_count < 1) {
            interval_count = 1;
        }
    }

    std::vector<GraphPoint> out;
    out.reserve(static_cast<std::size_t>(interval_count + 1));
    out.push_back(points_xy.front());

    std::size_t seg = 1;
    for (long long i = 1; i < interval_count; ++i) {
        const double target = total * static_cast<double>(i) / static_cast<double>(interval_count);
        while (seg < cumulative.size() && cumulative[seg] < target) {
            ++seg;
        }
        if (seg >= cumulative.size()) {
            out.push_back(points_xy.back());
            continue;
        }

        const double prev_len = cumulative[seg - 1];
        const double next_len = cumulative[seg];
        const double denom = next_len - prev_len;
        const double t = (denom > 0.0) ? ((target - prev_len) / denom) : 0.0;
        out.push_back(interpolate_point(points_xy[seg - 1], points_xy[seg], t));
    }

    out.push_back(points_xy.back());
    return out;
}

inline std::vector<GraphPoint> make_processed_path_points(
    const std::vector<int>& vertex_ids,
    const std::vector<GraphPoint>& vertices_xy,
    const std::vector<std::pair<double, double>>& polygon_xy,
    const GraphBuildOptions& options) {
    std::vector<GraphPoint> raw;
    raw.reserve(vertex_ids.size());
    for (int vertex_id : vertex_ids) {
        raw.push_back(vertices_xy[static_cast<std::size_t>(vertex_id)]);
    }

    if (raw.size() < 2) {
        return raw;
    }

    std::vector<GraphPoint> smoothed = chaikin_smooth_open(raw, options.smooth_iterations);
    std::vector<GraphPoint> resampled = resample_evenly(smoothed, options.resample_spacing);

    if (!resampled.empty()) {
        resampled.front() = raw.front();
        resampled.back() = raw.back();
    }

    if (options.enforce_within_polygon && polygon_xy.size() >= 3 && resampled.size() > 2) {
        for (std::size_t i = 1; i + 1 < resampled.size(); ++i) {
            if (!point_in_polygon_or_boundary(resampled[i], polygon_xy)) {
                resampled[i] = closest_point_on_polygon(resampled[i], polygon_xy);
            }
        }
    }

    return resampled;
}

inline PackedGraphData build_internal_graph_data(
    const std::vector<std::array<double, 4>>& ridges,
    const std::vector<std::pair<double, double>>& polygon_xy,
    const GraphBuildOptions& options) {
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
            vertex_to_node[v] = static_cast<int>(packed.nodes_xy.size());
            const auto& p = vertices_xy[v];
            packed.nodes_xy.push_back(std::array<double, 2>{p.x, p.y});
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
                graph_paths.push_back(GraphPath{node_u, node_v, path_vertices});
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
                const int synthetic_node = static_cast<int>(packed.nodes_xy.size());
                vertex_to_node[static_cast<std::size_t>(next_vertex)] = synthetic_node;
                is_node[static_cast<std::size_t>(next_vertex)] = 1;
                const auto& p = vertices_xy[static_cast<std::size_t>(next_vertex)];
                packed.nodes_xy.push_back(std::array<double, 2>{p.x, p.y});
                const int node_u = vertex_to_node[static_cast<std::size_t>(start_vertex)];
                graph_paths.push_back(GraphPath{node_u, synthetic_node, path_vertices});
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
            loop_node = static_cast<int>(packed.nodes_xy.size());
            vertex_to_node[static_cast<std::size_t>(start_vertex)] = loop_node;
            is_node[static_cast<std::size_t>(start_vertex)] = 1;
            const auto& p = vertices_xy[static_cast<std::size_t>(start_vertex)];
            packed.nodes_xy.push_back(std::array<double, 2>{p.x, p.y});
        }

        graph_paths.push_back(GraphPath{loop_node, loop_node, path_vertices});
    }

    packed.edge_offsets.reserve(graph_paths.size() + 1);
    packed.edge_offsets.push_back(0);

    for (const auto& path : graph_paths) {
        packed.edges_u.push_back(static_cast<long long>(path.node_u));
        packed.edges_v.push_back(static_cast<long long>(path.node_v));
        const auto processed = make_processed_path_points(path.vertices, vertices_xy, polygon_xy, options);
        packed.edge_weights.push_back(polyline_length(processed));

        for (const auto& p : processed) {
            packed.edge_points_xy.push_back(std::array<double, 2>{p.x, p.y});
        }

        packed.edge_offsets.push_back(static_cast<long long>(packed.edge_points_xy.size()));
    }

    return packed;
}

}
