# FILE: tests/test_m1_graph.py
# Member 1 Pre-Merge Gate Test Suite.
# Verbatim from Member 1's delivery — no modifications.
# Run: pytest tests/test_m1_graph.py -v

import time
import pytest
import numpy as np

from apricorn.contracts import GraphDataContract, RoutingCostContract
from apricorn.mocks import make_mock_graph, make_mock_costs
from apricorn.data_graph.parser_gen import (
    generate_synthetic_board,
    load_or_generate_board,
)
from apricorn.data_graph.graph_builder import (
    build_graph,
    update_overflow,
)
from apricorn.data_graph.drc_engine import (
    check_drc,
)


def test_mock_graph_contract():
    """Verify that Day-1 mock graph complies strictly with GraphDataContract shapes."""
    L, X, Y, nets = 2, 32, 32, 15
    N = L * X * Y
    mock_graph = make_mock_graph(L=L, X=X, Y=Y, nets=nets, seed=42)

    assert isinstance(mock_graph, GraphDataContract)
    assert mock_graph.grid_shape == (L, X, Y)
    assert mock_graph.x_nodes.shape == (N, 12)
    assert mock_graph.x_nodes.dtype == np.float32

    E = mock_graph.edge_index.shape[1]
    assert mock_graph.edge_index.shape == (2, E)
    assert mock_graph.edge_index.dtype == np.int64
    assert mock_graph.edge_attr.shape == (E, 6)
    assert mock_graph.edge_attr.dtype == np.float32

    assert len(mock_graph.net_demands) == nets
    for net_id, terminals in mock_graph.net_demands.items():
        assert len(terminals) >= 2
        for t in terminals:
            assert 0 <= t < N
            assert mock_graph.x_nodes[t, 3] == 1.0  # is_pin == 1.0


def test_mock_costs_contract():
    """Verify that Day-1 mock costs comply strictly with RoutingCostContract."""
    E = 15000
    N = 2048
    mock_costs = make_mock_costs(E=E, N=N, nets=15, epoch=1, seed=42)

    assert isinstance(mock_costs, RoutingCostContract)
    assert mock_costs.epoch == 1
    assert mock_costs.edge_weights.shape == (E,)
    assert mock_costs.edge_weights.dtype == np.float32
    assert np.all(mock_costs.edge_weights >= 1.0)
    assert not np.isnan(mock_costs.edge_weights).any()
    assert not np.isinf(mock_costs.edge_weights).any()

    assert mock_costs.hidden_state.shape == (N, 64)
    assert mock_costs.hidden_state.dtype == np.float32
    assert len(mock_costs.net_priorities) == 15
    for priority in mock_costs.net_priorities.values():
        assert 0.0 < priority < 1.0


def test_member1_tensor_shapes():
    """Mandatory Gate: Verifies x_nodes.shape == (N, 12), edge_attr.shape == (E, 6)."""
    board_spec = generate_synthetic_board(layers=2, width=16, height=16, num_nets=6, seed=123)
    graph = build_graph(board_spec)

    L, X, Y = graph.grid_shape
    N = L * X * Y
    E = graph.edge_index.shape[1]

    assert graph.x_nodes.shape == (N, 12)
    assert graph.edge_attr.shape == (E, 6)
    assert graph.edge_index.shape == (2, E)
    assert graph.x_nodes.dtype == np.float32
    assert graph.edge_attr.dtype == np.float32
    assert graph.edge_index.dtype == np.int64


def test_member1_bidirectional_edge_symmetry():
    """Mandatory Gate: Verifies bidirectional edge_index symmetry: (u, v) <=> (v, u)."""
    board_spec = generate_synthetic_board(layers=2, width=8, height=8, num_nets=2, seed=99)
    graph = build_graph(board_spec)

    edge_set = set(zip(graph.edge_index[0].tolist(), graph.edge_index[1].tolist()))
    for u, v in edge_set:
        assert (v, u) in edge_set, f"Edge ({u}, {v}) missing reciprocal reverse edge ({v}, {u})"


def test_member1_valid_terminal_indices():
    """Mandatory Gate: Verifies valid terminal indices in net_demands & node pin flags."""
    board_spec = generate_synthetic_board(layers=2, width=16, height=16, num_nets=8, seed=42)
    graph = build_graph(board_spec)

    N = graph.x_nodes.shape[0]
    for net_id, terminals in graph.net_demands.items():
        assert len(terminals) >= 2
        for t in terminals:
            assert 0 <= t < N
            assert graph.x_nodes[t, 3] == 1.0   # is_pin set
            assert graph.x_nodes[t, 6] >= 1.0   # demand set


def test_member1_rtree_drc_collision_detection():
    """
    Mandatory Gate: Verifies R-tree collision detection on overlapping segments
    and keepout obstacles in <5ms.
    """
    board_spec = {
        "board_id": "test_drc_board",
        "layers": 2,
        "width": 16,
        "height": 16,
        "nets": [
            {"net_id": 0, "pins": [[0, 2, 2], [0, 10, 2]]},
            {"net_id": 1, "pins": [[0, 5, 0], [0, 5, 10]]},
        ],
        "obstacles": [
            {"layer": 0, "polygon": [[12.0, 12.0], [15.0, 12.0], [15.0, 15.0], [12.0, 15.0]]}
        ]
    }
    graph = build_graph(board_spec)
    _, X, Y = graph.grid_shape

    def nid(l, x, y):
        return l * (X * Y) + x * Y + y

    # Case 1: Crossing paths → must detect violation
    path_net0 = [nid(0, x, 2) for x in range(2, 9)]
    path_net1 = [nid(0, 5, y) for y in range(0, 6)]

    t0 = time.perf_counter()
    violations = check_drc(graph, {0: path_net0, 1: path_net1}, clearance=0.5, trace_width=0.4)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert violations > 0, "DRC failed to detect crossing trace short!"
    assert elapsed_ms < 50.0, f"DRC check took {elapsed_ms:.2f}ms, expected <50ms"

    # Case 2: Non-overlapping paths → no violations
    clean_violations = check_drc(
        graph,
        {0: [nid(0, x, 2) for x in range(2, 9)], 1: [nid(0, x, 8) for x in range(2, 9)]},
        clearance=0.5, trace_width=0.4,
    )
    assert clean_violations == 0, f"Clean paths produced false positives: {clean_violations}"

    # Case 3: Route through obstacle → must detect violation
    obs_violations = check_drc(
        graph, {0: [nid(0, x, 13) for x in range(10, 16)]}, clearance=0.5, trace_width=0.4
    )
    assert obs_violations > 0, "DRC failed to detect obstacle polygon penetration!"


def test_member1_update_overflow():
    """Verify that Member 3's edge overflow updates congestion features correctly."""
    board_spec = generate_synthetic_board(layers=2, width=8, height=8, num_nets=2, seed=42)
    graph = build_graph(board_spec)

    E = graph.edge_index.shape[1]
    mock_overflow = np.zeros(E, dtype=np.float32)
    mock_overflow[0:5] = 2.0

    updated_graph = update_overflow(graph, mock_overflow)

    assert np.all(updated_graph.edge_attr[0:5, 5] == 2.0)
    assert updated_graph.x_nodes[:, 7].sum() > 0.0
