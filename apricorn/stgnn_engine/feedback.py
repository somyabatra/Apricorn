# FILE: apricorn/stgnn_engine/feedback.py
"""
Router -> graph feedback signal (numpy only; no torch needed).

The router reports `edge_overflow`, but on dense boards most DRC violations are
NODE shorts (two nets sharing a grid node), which produce no edge overflow at all.
If only edge_overflow is fed back, the STGNN is blind to the main failure mode.

`conflict_signal` returns one (E,) float32 vector combining both kinds of conflict.
Pass it to data_graph.graph_builder.update_overflow() exactly as you would
edge_overflow. It is used identically in training rollouts, evaluation, the
server and main.py, so the model sees the same kind of input everywhere.
"""
from __future__ import annotations

from typing import Dict, List, Set

import numpy as np

from apricorn.contracts import GraphDataContract, RoutedBoardResult


def collided_nodes(paths: Dict[int, List[int]]) -> Set[int]:
    """Nodes that appear in the routed path of more than one net."""
    owner: Dict[int, int] = {}
    hit: Set[int] = set()
    for nid, nodes in paths.items():
        for n in nodes:
            o = owner.setdefault(n, nid)
            if o != nid:
                hit.add(n)
    return hit


def conflict_signal(graph: GraphDataContract, result: RoutedBoardResult) -> np.ndarray:
    """(E,) float32: edge overflow + 1.0 for every edge touching a shared node."""
    sig = result.edge_overflow.astype(np.float32).copy()
    hit = collided_nodes(result.routed_paths)
    if hit:
        N = graph.x_nodes.shape[0]
        mask = np.zeros(N, dtype=bool)
        mask[list(hit)] = True
        src, dst = graph.edge_index
        sig += (mask[src] | mask[dst]).astype(np.float32)
    return sig
