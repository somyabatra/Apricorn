"""
Project Apricorn - apricorn.data_graph
Subpackage owned by Member 1 (Graph & DRC Architect).
Responsible for:
- Board JSON/KiCad parsing and synthetic dataset generation (parser_gen.py)
- 3D Octilinear Grid-Graph Extractor (graph_builder.py)
- Pure-Software Geometric DRC Engine (drc_engine.py)
"""

from apricorn.data_graph.parser_gen import (
    load_or_generate_board,
    generate_synthetic_board,
    generate_synthetic_dataset,
)
from apricorn.data_graph.graph_builder import (
    build_graph,
    update_overflow,
)
from apricorn.data_graph.drc_engine import (
    check_drc,
)

__all__ = [
    "load_or_generate_board",
    "generate_synthetic_board",
    "generate_synthetic_dataset",
    "build_graph",
    "update_overflow",
    "check_drc",
]
