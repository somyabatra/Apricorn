# FILE: apricorn/stgnn_engine/heads.py
# Member 2 — Dual Decoding Heads
#
# Head 1 — Edge Cost Head:
#   Input:  [H_t[src] || H_t[dst] || edge_attr]  → 134 dims
#   Output: edge_weights (E,) float32  ALL >= 1.0  via  1 + Softplus(x)
#
# Head 2 — Net Priority Head:
#   Input:  mean-pooled terminal embeddings per net
#   Output: Dict[net_id → priority in (0.0, 1.0)]  via Sigmoid

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

HIDDEN_DIM = 64
EDGE_FEAT_DIM = 6
EDGE_MLP_INPUT = HIDDEN_DIM + HIDDEN_DIM + EDGE_FEAT_DIM  # 134


class EdgeCostHead(nn.Module):
    """
    Predicts a positive traversal cost for every directed edge.

    Architecture:
        [H_t[src] || H_t[dst] || edge_attr]   (134,)
            → Linear(134, 64) → ELU
            → Linear(64, 32)  → ELU
            → Linear(32, 1)
            → 1.0 + Softplus(·)               ← guarantees output >= 1.0

    The Softplus activation (log(1 + exp(x))) is strictly positive, so
    adding 1.0 guarantees edge_weights > 1.0 with no clipping needed.
    """

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        edge_dim: int = EDGE_FEAT_DIM,
        mid_dim: int = 64,
    ) -> None:
        super().__init__()
        in_dim = hidden_dim * 2 + edge_dim  # 134

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.ELU(),
            nn.Linear(mid_dim, 32),
            nn.ELU(),
            nn.Linear(32, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                nn.init.zeros_(m.bias)

    def forward(
        self,
        h_t: Tensor,          # (N, HIDDEN_DIM)
        edge_index: Tensor,   # (2, E)  int64
        edge_attr: Tensor,    # (E, EDGE_FEAT_DIM)
    ) -> Tensor:              # (E,) float32  — all values >= 1.0
        src, dst = edge_index[0], edge_index[1]

        # Gather source and destination node embeddings
        h_src = h_t[src]      # (E, HIDDEN_DIM)
        h_dst = h_t[dst]      # (E, HIDDEN_DIM)

        # Concatenate: [h_src || h_dst || edge_attr]  → (E, 134)
        edge_input = torch.cat([h_src, h_dst, edge_attr], dim=-1)

        raw = self.mlp(edge_input).squeeze(-1)        # (E,)
        edge_weights = 1.0 + F.softplus(raw)          # (E,)  strictly >= 1.0
        return edge_weights


class NetPriorityHead(nn.Module):
    """
    Predicts a routing priority score in (0, 1) for each net.

    For each net, the terminal node embeddings are mean-pooled, then passed
    through a small MLP + Sigmoid to produce the priority score.

    Higher score => Member 3 routes that net first.
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ELU(),
            nn.Linear(32, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                nn.init.zeros_(m.bias)

    def forward_tensor(
        self,
        h_t: Tensor,                         # (N, HIDDEN_DIM)
        net_demands: Dict[int, List[int]],   # {net_id: [terminal node indices]}
    ) -> "tuple[List[int], Tensor]":
        """
        Differentiable variant used for TRAINING. Returns (net_ids, scores) where
        scores has shape (K,) in (0, 1) and carries gradients back into the model.
        (forward() below returns python floats via .item(), which cuts the graph.)
        """
        net_ids = [nid for nid, t in net_demands.items() if t]
        if not net_ids:
            return [], h_t.new_zeros(0)
        pooled = torch.stack(
            [h_t[torch.as_tensor(net_demands[nid], dtype=torch.long, device=h_t.device)].mean(dim=0)
             for nid in net_ids],
            dim=0,
        )                                                    # (K, HIDDEN_DIM)
        scores = torch.sigmoid(self.mlp(pooled)).squeeze(-1)  # (K,)
        return net_ids, scores

    def forward(
        self,
        h_t: Tensor,                         # (N, HIDDEN_DIM)
        net_demands: Dict[int, List[int]],   # {net_id: [terminal node indices]}
        device: torch.device,
    ) -> Dict[int, float]:                   # {net_id: priority in (0, 1)}
        net_priorities: Dict[int, float] = {}

        for net_id, terminal_indices in net_demands.items():
            if not terminal_indices:
                net_priorities[net_id] = 0.5
                continue

            idx = torch.tensor(terminal_indices, dtype=torch.long, device=device)
            terminal_embeddings = h_t[idx]               # (k, HIDDEN_DIM)
            pooled = terminal_embeddings.mean(dim=0, keepdim=True)  # (1, HIDDEN_DIM)

            raw_score = self.mlp(pooled)                 # (1, 1)
            priority = torch.sigmoid(raw_score).item()   # scalar in (0, 1)
            net_priorities[net_id] = float(priority)

        return net_priorities


class ApricornHeads(nn.Module):
    """
    Convenience wrapper that holds both decoding heads.
    Used by predictor.py for clean checkpoint save/load.
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM, edge_dim: int = EDGE_FEAT_DIM) -> None:
        super().__init__()
        self.edge_cost_head = EdgeCostHead(hidden_dim=hidden_dim, edge_dim=edge_dim)
        self.net_priority_head = NetPriorityHead(hidden_dim=hidden_dim)
