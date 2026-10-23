# FILE: apricorn/contracts.py
# LOCKED DAY-0 — DO NOT MODIFY WITHOUT ALL 3 MEMBERS AGREEING.
# This is the single immutable mathematical handshake between all three members.
# Source: Member 1's canonical version (authoritative).

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class GraphDataContract:
    """
    Produced by MEMBER 1 (data_graph) -> Consumed by MEMBER 2 (stgnn) & MEMBER 3 (router_ui).

    x_nodes column index reference (12 features):
        0: x            — grid x-coordinate (float)
        1: y            — grid y-coordinate (float)
        2: z            — layer index (float)
        3: is_pin       — 1.0 if node is a component pin, else 0.0
        4: is_obs       — 1.0 if node is blocked by an obstacle, else 0.0
        5: cap          — routing capacity (normalised)
        6: dem          — routing demand (>= 1.0 at pin nodes)
        7: h_overflow   — historical congestion overflow from previous epochs
        8: net_id_norm  — normalised net identifier (-1.0 if no net)
        9: deg          — node degree in the grid graph (float)
       10: dist_bnd     — distance to nearest board boundary (float)
       11: layer_dir    — preferred routing direction: 0=Horizontal, 1=Vertical

    edge_attr column index reference (6 features):
        0: length           — Euclidean length (1.0 ortho, 1.4142 diag, 1.0 via)
        1: dx               — signed delta-x
        2: dy               — signed delta-y
        3: dz_via           — dl (±1) if inter-layer via edge, else 0.0
        4: base_cost        — static routing cost (1.0 ortho, 1.4142 diag, 4.0 via)
        5: congestion_ratio — current epoch congestion; updated by update_overflow()
    """

    board_id: str
    grid_shape: Tuple[int, int, int]    # (Layers L, Grid_X, Grid_Y)

    x_nodes: np.ndarray                 # Shape: (N, 12)  float32
    edge_index: np.ndarray              # Shape: (2, E)   int64  — (src, dst) directed pairs
    edge_attr: np.ndarray               # Shape: (E, 6)   float32

    net_demands: Dict[int, List[int]]   # {net_id: [ordered terminal node indices]}
    obstacle_polygons: List[Tuple[int, List[Tuple[float, float]]]]  # [(layer, [(x,y), ...])]


@dataclass(frozen=True)
class RoutingCostContract:
    """
    Produced by MEMBER 2 (stgnn_engine) -> Consumed by MEMBER 3 (router_ui).

    Guarantees (enforced by 1.0 + Softplus in heads.py):
      - edge_weights: every element strictly >= 1.0, shape (E,) float32
      - net_priorities: values in (0.0, 1.0), higher score => routed first by M3
      - hidden_state: GRU state H_t fed back as prev_hidden on epoch t+1
    """

    epoch: int
    edge_weights: np.ndarray           # Shape: (E,)    float32  — all values >= 1.0
    net_priorities: Dict[int, float]   # {net_id: priority score in (0.0, 1.0)}
    hidden_state: np.ndarray           # Shape: (N, 64) float32  — persistent GRU state H_t


@dataclass(frozen=True)
class RoutedBoardResult:
    """
    Produced by MEMBER 3 (router_ui) -> Consumed by Web UI & Member 2 feedback loop.
    """

    epoch: int
    routed_paths: Dict[int, List[int]]  # {net_id: [node_index sequence of routed trace]}
    edge_overflow: np.ndarray           # Shape: (E,) float32 — fed back to M1/M2 for t+1
    drc_violations: int                 # 0 => fully converged
    wirelength_mm: float                # Total octilinear trace length
    via_count: int                      # Total inter-layer Z-transitions
