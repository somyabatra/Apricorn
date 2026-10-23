# FILE: apricorn/stgnn_engine/model.py
# Member 2 — STGNN Core Neural Network
#
# Architecture:
#   GraphDataContract.x_nodes (N,12) + edge_index (2,E) + edge_attr (E,6)
#       ↓
#   GATv2Conv × 3   (12→64, 64→64, 64→64 | 4 heads | edge_dim=6)
#       ↓
#   GraphConvGRUCell  H_{t-1}(N,64) + H_spatial(N,64) → H_t(N,64)
#       ↓
#   H_t  ─── Edge Cost Head  →  edge_weights (E,)   [>= 1.0]
#        └── Net Priority Head → net_priorities dict  [(0,1)]

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATv2Conv


# ---------------------------------------------------------------------------
# Constants matching contracts.py
# ---------------------------------------------------------------------------
NODE_FEAT_DIM = 12   # x_nodes columns
EDGE_FEAT_DIM = 6    # edge_attr columns
HIDDEN_DIM = 64      # GRU / embedding dimension
GAT_HEADS = 4        # number of attention heads (output is head-averaged)
GAT_LAYERS = 3       # depth of spatial encoder


class GATv2SpatialEncoder(nn.Module):
    """
    3-layer GATv2Conv spatial encoder.

    Each layer uses multi-head attention conditioned on edge features.
    Heads are averaged (concat=False) to keep the hidden dim constant at HIDDEN_DIM.
    A residual projection is added from layer 2 onward.
    """

    def __init__(
        self,
        in_channels: int = NODE_FEAT_DIM,
        hidden: int = HIDDEN_DIM,
        heads: int = GAT_HEADS,
        edge_dim: int = EDGE_FEAT_DIM,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = dropout

        # Layer 0: in_channels → hidden
        self.conv0 = GATv2Conv(
            in_channels=in_channels,
            out_channels=hidden,
            heads=heads,
            concat=False,      # output dim = hidden (not heads*hidden)
            edge_dim=edge_dim,
            dropout=dropout,
            add_self_loops=False,
        )
        self.norm0 = nn.LayerNorm(hidden)

        # Layer 1: hidden → hidden (with residual)
        self.conv1 = GATv2Conv(
            in_channels=hidden,
            out_channels=hidden,
            heads=heads,
            concat=False,
            edge_dim=edge_dim,
            dropout=dropout,
            add_self_loops=False,
        )
        self.norm1 = nn.LayerNorm(hidden)

        # Layer 2: hidden → hidden (with residual)
        self.conv2 = GATv2Conv(
            in_channels=hidden,
            out_channels=hidden,
            heads=heads,
            concat=False,
            edge_dim=edge_dim,
            dropout=dropout,
            add_self_loops=False,
        )
        self.norm2 = nn.LayerNorm(hidden)

    def forward(
        self,
        x: Tensor,           # (N, NODE_FEAT_DIM)
        edge_index: Tensor,  # (2, E)  int64
        edge_attr: Tensor,   # (E, EDGE_FEAT_DIM)
    ) -> Tensor:             # (N, HIDDEN_DIM)
        # Layer 0
        h = F.elu(self.norm0(self.conv0(x, edge_index, edge_attr)))
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 1 + residual
        h = F.elu(self.norm1(self.conv1(h, edge_index, edge_attr))) + h
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 2 + residual
        h = F.elu(self.norm2(self.conv2(h, edge_index, edge_attr))) + h

        return h  # (N, HIDDEN_DIM)


class GraphConvGRUCell(nn.Module):
    """
    Graph-aware GRU cell that updates the temporal hidden state H_t.

    Instead of a vanilla linear gate, the reset/update/candidate gates each
    use a single linear projection over [H_{t-1} || H_spatial].  This keeps
    the implementation graph-agnostic (no additional message-passing inside
    the GRU itself) while still being conditioned on the spatial context.

    H_t = GRU(H_{t-1}, H_spatial)
    """

    def __init__(self, input_dim: int = HIDDEN_DIM, hidden_dim: int = HIDDEN_DIM) -> None:
        super().__init__()
        combined = input_dim + hidden_dim

        # Reset gate
        self.W_r = nn.Linear(combined, hidden_dim, bias=True)
        # Update gate
        self.W_z = nn.Linear(combined, hidden_dim, bias=True)
        # Candidate hidden state
        self.W_n = nn.Linear(combined, hidden_dim, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        for layer in [self.W_r, self.W_z, self.W_n]:
            nn.init.orthogonal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(
        self,
        h_spatial: Tensor,   # (N, HIDDEN_DIM) — output of GATv2 encoder at epoch t
        h_prev: Tensor,      # (N, HIDDEN_DIM) — H_{t-1}; zeros on epoch 1
    ) -> Tensor:             # (N, HIDDEN_DIM) — H_t
        combined = torch.cat([h_spatial, h_prev], dim=-1)  # (N, 2*HIDDEN_DIM)

        r = torch.sigmoid(self.W_r(combined))               # reset gate
        z = torch.sigmoid(self.W_z(combined))               # update gate

        # Candidate uses reset gate to modulate previous hidden
        combined_reset = torch.cat([h_spatial, r * h_prev], dim=-1)
        n = torch.tanh(self.W_n(combined_reset))             # candidate

        h_t = (1.0 - z) * h_prev + z * n                   # (N, HIDDEN_DIM)
        return h_t


class ApricornSTGNN(nn.Module):
    """
    Full Spatio-Temporal GNN model for Project Apricorn.

    Forward pass:
        x_nodes, edge_index, edge_attr, h_prev  →  H_t

    The decoding heads (edge cost + net priority) live in heads.py and
    are composed in predictor.py to keep this module single-responsibility.
    """

    def __init__(
        self,
        node_feat_dim: int = NODE_FEAT_DIM,
        edge_feat_dim: int = EDGE_FEAT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        gat_heads: int = GAT_HEADS,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.spatial_encoder = GATv2SpatialEncoder(
            in_channels=node_feat_dim,
            hidden=hidden_dim,
            heads=gat_heads,
            edge_dim=edge_feat_dim,
            dropout=dropout,
        )
        self.gru_cell = GraphConvGRUCell(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
        )

    def forward(
        self,
        x: Tensor,           # (N, 12)
        edge_index: Tensor,  # (2, E)
        edge_attr: Tensor,   # (E, 6)
        h_prev: Tensor,      # (N, 64) — zeros on first epoch
    ) -> Tensor:             # H_t: (N, 64)
        h_spatial = self.spatial_encoder(x, edge_index, edge_attr)  # (N, 64)
        h_t = self.gru_cell(h_spatial, h_prev)                      # (N, 64)
        return h_t

    def init_hidden(self, N: int, device: torch.device) -> Tensor:
        """Return a zero hidden state for epoch 1."""
        return torch.zeros(N, self.hidden_dim, device=device)
