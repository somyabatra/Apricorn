# FILE: tests/test_m2_stgnn.py
# Member 2 Pre-Merge Gate Tests
#
# Run: pytest tests/test_m2_stgnn.py -v
#
# ALL assertions below must pass before opening a PR to main.

from __future__ import annotations

import numpy as np
import pytest
import torch

from apricorn.mocks import make_mock_graph, make_mock_costs
from apricorn.stgnn_engine.predictor import STGNNPredictor
from apricorn.stgnn_engine.model import ApricornSTGNN, GraphConvGRUCell, GATv2SpatialEncoder
from apricorn.stgnn_engine.heads import EdgeCostHead, NetPriorityHead

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mock_graph():
    """Standard 2-layer 32×32 board with 15 nets."""
    return make_mock_graph(L=2, X=32, Y=32, nets=15, seed=42)


@pytest.fixture(scope="module")
def predictor():
    """Untrained predictor (no checkpoint needed for shape/contract tests)."""
    return STGNNPredictor(checkpoint_path="checkpoints/stgnn_v1.pt")


@pytest.fixture(scope="module")
def device():
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Contract shape tests
# ---------------------------------------------------------------------------

class TestOutputContracts:
    """Primary pre-merge gate — must ALL pass."""

    def test_edge_weights_shape(self, predictor, mock_graph):
        costs = predictor.predict(mock_graph, prev_hidden=None, epoch=1)
        E = mock_graph.edge_index.shape[1]
        assert costs.edge_weights.shape == (E,), (
            f"edge_weights.shape={costs.edge_weights.shape}, expected ({E},)"
        )

    def test_edge_weights_all_gte_1(self, predictor, mock_graph):
        costs = predictor.predict(mock_graph, prev_hidden=None, epoch=1)
        assert np.all(costs.edge_weights >= 1.0), (
            f"Minimum edge_weight={costs.edge_weights.min():.6f} < 1.0"
        )

    def test_edge_weights_dtype(self, predictor, mock_graph):
        costs = predictor.predict(mock_graph, prev_hidden=None, epoch=1)
        assert costs.edge_weights.dtype == np.float32

    def test_edge_weights_no_nan(self, predictor, mock_graph):
        costs = predictor.predict(mock_graph, prev_hidden=None, epoch=1)
        assert not np.any(np.isnan(costs.edge_weights)), "NaN found in edge_weights"

    def test_edge_weights_no_inf(self, predictor, mock_graph):
        costs = predictor.predict(mock_graph, prev_hidden=None, epoch=1)
        assert not np.any(np.isinf(costs.edge_weights)), "Inf found in edge_weights"

    def test_hidden_state_shape(self, predictor, mock_graph):
        costs = predictor.predict(mock_graph, prev_hidden=None, epoch=1)
        N = mock_graph.x_nodes.shape[0]
        assert costs.hidden_state.shape == (N, 64), (
            f"hidden_state.shape={costs.hidden_state.shape}, expected ({N}, 64)"
        )

    def test_hidden_state_dtype(self, predictor, mock_graph):
        costs = predictor.predict(mock_graph, prev_hidden=None, epoch=1)
        assert costs.hidden_state.dtype == np.float32

    def test_hidden_state_no_nan(self, predictor, mock_graph):
        costs = predictor.predict(mock_graph, prev_hidden=None, epoch=1)
        assert not np.any(np.isnan(costs.hidden_state))

    def test_hidden_state_no_inf(self, predictor, mock_graph):
        costs = predictor.predict(mock_graph, prev_hidden=None, epoch=1)
        assert not np.any(np.isinf(costs.hidden_state))

    def test_net_priorities_keys_match(self, predictor, mock_graph):
        costs = predictor.predict(mock_graph, prev_hidden=None, epoch=1)
        assert set(costs.net_priorities.keys()) == set(mock_graph.net_demands.keys())

    def test_net_priorities_in_range(self, predictor, mock_graph):
        costs = predictor.predict(mock_graph, prev_hidden=None, epoch=1)
        for net_id, priority in costs.net_priorities.items():
            assert 0.0 < priority < 1.0, (
                f"net {net_id} priority={priority:.4f} not in (0, 1)"
            )

    def test_epoch_field(self, predictor, mock_graph):
        costs = predictor.predict(mock_graph, prev_hidden=None, epoch=7)
        assert costs.epoch == 7


# ---------------------------------------------------------------------------
# Temporal continuity: pass hidden_state across epochs
# ---------------------------------------------------------------------------

class TestTemporalContinuity:

    def test_hidden_state_changes_across_epochs(self, predictor, mock_graph):
        """H_t should evolve — two consecutive epochs must NOT produce identical states."""
        costs_1 = predictor.predict(mock_graph, prev_hidden=None, epoch=1)
        costs_2 = predictor.predict(mock_graph, prev_hidden=costs_1.hidden_state, epoch=2)
        assert not np.allclose(costs_1.hidden_state, costs_2.hidden_state), (
            "H_1 == H_2: GRU is not updating the hidden state"
        )

    def test_hidden_state_passthrough_shape_preserved(self, predictor, mock_graph):
        """Hidden state shape (N, 64) must be preserved across 5 epochs."""
        N = mock_graph.x_nodes.shape[0]
        h = None
        for epoch in range(1, 6):
            costs = predictor.predict(mock_graph, prev_hidden=h, epoch=epoch)
            h = costs.hidden_state
            assert h.shape == (N, 64), f"Epoch {epoch}: hidden shape {h.shape} != ({N}, 64)"


# ---------------------------------------------------------------------------
# Internal module shape tests
# ---------------------------------------------------------------------------

class TestModelInternals:

    def test_gat_encoder_output_shape(self, mock_graph, device):
        model = GATv2SpatialEncoder()
        model.eval()
        x = torch.from_numpy(mock_graph.x_nodes)
        ei = torch.from_numpy(mock_graph.edge_index)
        ea = torch.from_numpy(mock_graph.edge_attr)
        with torch.no_grad():
            out = model(x, ei, ea)
        N = mock_graph.x_nodes.shape[0]
        assert out.shape == (N, 64), f"GATv2 output shape {out.shape}"

    def test_gru_cell_output_shape(self, mock_graph, device):
        N = mock_graph.x_nodes.shape[0]
        cell = GraphConvGRUCell()
        cell.eval()
        h_s = torch.zeros(N, 64)
        h_p = torch.zeros(N, 64)
        with torch.no_grad():
            h_t = cell(h_s, h_p)
        assert h_t.shape == (N, 64)

    def test_edge_cost_head_output_shape(self, mock_graph, device):
        head = EdgeCostHead()
        head.eval()
        N = mock_graph.x_nodes.shape[0]
        E = mock_graph.edge_index.shape[1]
        h_t = torch.zeros(N, 64)
        ei = torch.from_numpy(mock_graph.edge_index)
        ea = torch.from_numpy(mock_graph.edge_attr)
        with torch.no_grad():
            ew = head(h_t, ei, ea)
        assert ew.shape == (E,)
        assert torch.all(ew >= 1.0), f"Min edge weight {ew.min().item():.6f} < 1.0"

    def test_net_priority_head_output(self, mock_graph, device):
        head = NetPriorityHead()
        head.eval()
        N = mock_graph.x_nodes.shape[0]
        h_t = torch.zeros(N, 64)
        with torch.no_grad():
            priorities = head(h_t, mock_graph.net_demands, device)
        for net_id, p in priorities.items():
            assert 0.0 < p < 1.0, f"net {net_id} priority={p}"


# ---------------------------------------------------------------------------
# Edge-case: tiny 1-layer board
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_single_layer_board(self, predictor):
        tiny = make_mock_graph(L=1, X=8, Y=8, nets=3, seed=99)
        costs = predictor.predict(tiny, prev_hidden=None, epoch=1)
        E = tiny.edge_index.shape[1]
        N = tiny.x_nodes.shape[0]
        assert costs.edge_weights.shape == (E,)
        assert costs.hidden_state.shape == (N, 64)
        assert np.all(costs.edge_weights >= 1.0)

    def test_large_board(self, predictor):
        large = make_mock_graph(L=4, X=32, Y=32, nets=30, seed=7)
        costs = predictor.predict(large, prev_hidden=None, epoch=1)
        E = large.edge_index.shape[1]
        N = large.x_nodes.shape[0]
        assert costs.edge_weights.shape == (E,)
        assert costs.hidden_state.shape == (N, 64)

    def test_wrong_hidden_shape_raises(self, predictor, mock_graph):
        bad_hidden = np.zeros((5, 64), dtype=np.float32)  # wrong N
        with pytest.raises((ValueError, AssertionError)):
            predictor.predict(mock_graph, prev_hidden=bad_hidden, epoch=2)
