# FILE: apricorn/stgnn_engine/predictor.py
# Member 2 — Public Inference API
#
# This is the ONLY class Member 3 ever calls.
# Contract: STGNNPredictor.predict(graph, prev_hidden, epoch) -> RoutingCostContract
#
# Guarantees (enforced here + in heads.py):
#   - edge_weights.shape  == (E,)   float32
#   - np.all(edge_weights >= 1.0)   (1 + Softplus)
#   - hidden_state.shape  == (N, 64) float32
#   - zero NaN / Inf in any output tensor

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from apricorn.contracts import GraphDataContract, RoutingCostContract
from apricorn.stgnn_engine.model import ApricornSTGNN
from apricorn.stgnn_engine.heads import ApricornHeads

# Default checkpoint path (relative to project root)
DEFAULT_CHECKPOINT = os.path.join("checkpoints", "stgnn_v1.pt")


class STGNNPredictor:
    """
    Stateless inference wrapper for the Apricorn STGNN.

    Usage (Member 3 perspective):
        predictor = STGNNPredictor(checkpoint_path="checkpoints/stgnn_v1.pt")
        hidden = None
        for epoch in range(1, 21):
            costs = predictor.predict(graph, hidden, epoch)
            hidden = costs.hidden_state   # feed back into next epoch
    """

    def __init__(
        self,
        checkpoint_path: str = DEFAULT_CHECKPOINT,
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = ApricornSTGNN().to(self.device)
        self.heads = ApricornHeads().to(self.device)

        self.checkpoint_loaded = False
        ckpt_path = Path(checkpoint_path)
        if ckpt_path.exists():
            self._load_checkpoint(ckpt_path)
            self.checkpoint_loaded = True
        else:
            # During Day-1 development, no checkpoint exists yet — run untrained
            print(
                f"[STGNNPredictor] WARNING: checkpoint not found at '{checkpoint_path}'. "
                "Running with random (untrained) weights. Run train.py to create a checkpoint."
            )

        self.model.eval()
        self.heads.eval()

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def _load_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.heads.load_state_dict(ckpt["heads_state_dict"])
        epoch = ckpt.get("epoch", "?")
        loss = ckpt.get("loss", float("nan"))
        print(f"[STGNNPredictor] Loaded checkpoint '{path}' (epoch={epoch}, loss={loss:.4f})")

    def save_checkpoint(self, path: str = DEFAULT_CHECKPOINT) -> None:
        """Save current model weights (useful after fine-tuning)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "heads_state_dict": self.heads.state_dict(),
                "epoch": 0,
                "loss": float("nan"),
                "seed": 42,
            },
            path,
        )
        print(f"[STGNNPredictor] Saved checkpoint to '{path}'")

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        graph: GraphDataContract,
        prev_hidden: Optional[np.ndarray],  # (N, 64) float32 or None (epoch 1)
        epoch: int,
    ) -> RoutingCostContract:
        """
        Run one STGNN inference pass and return a locked RoutingCostContract.

        Args:
            graph:       GraphDataContract produced by Member 1 (or mocks.py).
            prev_hidden: H_{t-1} numpy array of shape (N, 64), or None on epoch 1.
            epoch:       Current negotiation epoch index (1-based).

        Returns:
            A frozen RoutingCostContract with:
              - edge_weights (E,) float32  — all values >= 1.0
              - net_priorities Dict[int, float]  — values in (0, 1)
              - hidden_state  (N, 64) float32    — for next epoch's prev_hidden
        """
        # --- Convert graph arrays to tensors ---
        x = torch.from_numpy(graph.x_nodes).to(self.device)           # (N, 12)
        edge_index = torch.from_numpy(graph.edge_index).to(self.device)  # (2, E) int64
        edge_attr = torch.from_numpy(graph.edge_attr).to(self.device)   # (E, 6)
        N = x.shape[0]

        # --- Initialise or restore hidden state ---
        if prev_hidden is None:
            h_prev = self.model.init_hidden(N, self.device)  # zeros
        else:
            if prev_hidden.shape != (N, 64):
                raise ValueError(
                    f"prev_hidden shape mismatch: expected ({N}, 64), got {prev_hidden.shape}"
                )
            h_prev = torch.from_numpy(prev_hidden).to(self.device)

        # --- Forward pass ---
        h_t = self.model(x, edge_index, edge_attr, h_prev)              # (N, 64)

        edge_weights_t = self.heads.edge_cost_head(h_t, edge_index, edge_attr)   # (E,)
        net_priorities = self.heads.net_priority_head(h_t, graph.net_demands, self.device)

        # --- Convert to numpy + validate ---
        edge_weights_np = edge_weights_t.cpu().numpy().astype(np.float32)
        hidden_np = h_t.cpu().numpy().astype(np.float32)

        self._validate_outputs(edge_weights_np, hidden_np, graph)

        return RoutingCostContract(
            epoch=epoch,
            edge_weights=edge_weights_np,
            net_priorities=net_priorities,
            hidden_state=hidden_np,
        )

    # ------------------------------------------------------------------
    # Output validation (pre-merge gate helper)
    # ------------------------------------------------------------------

    def _validate_outputs(
        self,
        edge_weights: np.ndarray,
        hidden_state: np.ndarray,
        graph: GraphDataContract,
    ) -> None:
        """
        Raises AssertionError if any contract guarantee is violated.
        These checks mirror tests/test_m2_stgnn.py exactly.
        """
        E = graph.edge_index.shape[1]
        N = graph.x_nodes.shape[0]

        assert edge_weights.shape == (E,), (
            f"edge_weights shape {edge_weights.shape} != ({E},)"
        )
        assert np.all(edge_weights >= 1.0), (
            f"edge_weights minimum {edge_weights.min():.6f} < 1.0"
        )
        assert not np.any(np.isnan(edge_weights)), "NaN detected in edge_weights"
        assert not np.any(np.isinf(edge_weights)), "Inf detected in edge_weights"

        assert hidden_state.shape == (N, 64), (
            f"hidden_state shape {hidden_state.shape} != ({N}, 64)"
        )
        assert not np.any(np.isnan(hidden_state)), "NaN detected in hidden_state"
        assert not np.any(np.isinf(hidden_state)), "Inf detected in hidden_state"
