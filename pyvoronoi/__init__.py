"""pyvoronoi -- Python wrapper for the Boost Voronoi library."""

from pyvoronoi import module as _module
from pyvoronoi.model import Cell, Edge, Vertex, VoronoiException
from pyvoronoi.module import (
    Point,
    PyVoronoi,
    Segment,
    generate_internal_graph,
    generate_internal_segments,
)


def __getattr__(name: str):
    if name == "SILENT":
        return _module.SILENT
    raise AttributeError(f"module 'pyvoronoi' has no attribute {name!r}")


def __setattr__(name: str, value):
    if name == "SILENT":
        _module.SILENT = value
        return
    raise AttributeError(f"module 'pyvoronoi' has no attribute {name!r}")


__all__ = [
    "Cell",
    "Edge",
    "Point",
    "PyVoronoi",
    "Segment",
    "SILENT",
    "Vertex",
    "VoronoiException",
    "generate_internal_graph",
    "generate_internal_segments",
]
