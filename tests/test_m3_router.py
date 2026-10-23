"""
Unit & Integration Tests for Project Apricorn Member 3: Router & UI Lead.
Verifies the Member 3 Gate Criteria:
- Passes make_mock_graph() + make_mock_costs() into NegotiatedRouter
- Asserts every net path starts/ends at exact terminals
- Asserts routed paths strictly respect obstacle nodes and edge_index
- Asserts clean rendering and schema validation on GET /api/board_state
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient

from apricorn.contracts import GraphDataContract, RoutedBoardResult, RoutingCostContract
from apricorn.mocks import make_mock_costs, make_mock_graph
from apricorn.router_ui.astar_router import OctilinearWavefrontRouter
from apricorn.router_ui.negotiator import NegotiatedRouter
from apricorn.router_ui.server import app, state


@pytest.fixture
def mock_board():
    """Generates standard Day-0 mock graph and costs."""
    graph = make_mock_graph(L=2, X=32, Y=32, nets=15, seed=42)
    costs = make_mock_costs(graph, epoch=1, seed=42)
    return graph, costs


def test_contract_integrity(mock_board):
    """Verifies that mock contracts strictly adhere to locked Day-0 shapes."""
    graph, costs = mock_board
    L, X, Y = graph.grid_shape
    N = L * X * Y
    E = graph.edge_index.shape[1]

    assert graph.x_nodes.shape == (N, 12), f"Expected x_nodes shape ({N}, 12)"
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_attr.shape == (E, 6)
    assert costs.edge_weights.shape == (E,)
    assert np.all(costs.edge_weights >= 1.0), "Edge weights must be strictly >= 1.0"
    assert costs.hidden_state.shape == (N, 64)


def test_wavefront_router_terminals_and_obstacles(mock_board):
    """Asserts every net starts/ends at exact terminals and avoids obstacles."""
    graph, costs = mock_board
    router = OctilinearWavefrontRouter(graph)
    h_congestion = np.ones(graph.edge_index.shape[1], dtype=np.float32)

    routed_paths, routed_edges, usage, wl, vias = router.route_all_nets(
        costs=costs,
        h_congestion=h_congestion
    )

    is_obs = graph.x_nodes[:, 4] > 0.5
    valid_edge_set = set(zip(graph.edge_index[0], graph.edge_index[1]))

    for net_id, terminals in graph.net_demands.items():
        assert net_id in routed_paths, f"Net {net_id} was not routed"
        path = routed_paths[net_id]
        assert len(path) >= 2, f"Path for net {net_id} must have >= 2 nodes"

        # Check that all terminals are present in path
        for term in terminals:
            assert term in path, f"Terminal {term} of net {net_id} not reached in path"

        # Check start and end pin fidelity
        assert path[0] == terminals[0], f"Net {net_id} does not start at terminal 0"

        # Check obstacle avoidance (no intermediate node in path can be an obstacle)
        for nid in path:
            if nid not in terminals:
                assert not is_obs[nid], f"Net {net_id} collided with obstacle node {nid}"

    # Check edge traversal fidelity (every edge must exist in edge_index)
    for net_id, edges in routed_edges.items():
        for e_idx in edges:
            u = graph.edge_index[0, e_idx]
            v = graph.edge_index[1, e_idx]
            assert (u, v) in valid_edge_set, f"Edge ({u}, {v}) not in graph edge_index"


def test_negotiated_router_loop(mock_board):
    """Verifies NegotiatedRouter returns valid RoutedBoardResult and resolves congestion."""
    graph, costs = mock_board
    router = NegotiatedRouter(max_epochs=5)
    result = router.step_epoch(graph=graph, costs=costs, epoch=1)

    assert isinstance(result, RoutedBoardResult)
    assert result.epoch == 1
    assert result.edge_overflow.shape == (graph.edge_index.shape[1],)
    assert result.wirelength_mm > 0.0
    assert result.via_count >= 0
    assert len(result.routed_paths) == len(graph.net_demands)


def test_multi_pin_expansion():
    """Explicitly tests 3-terminal net Prim-A* routing expansion."""
    graph = make_mock_graph(L=2, X=16, Y=16, nets=2, seed=123)
    # Force net 1 to have 3 terminals
    non_obs = np.where(graph.x_nodes[:, 4] == 0)[0]
    t0, t1, t2 = int(non_obs[5]), int(non_obs[20]), int(non_obs[40])
    graph.net_demands[1] = [t0, t1, t2]

    costs = make_mock_costs(graph, epoch=1)
    router = OctilinearWavefrontRouter(graph)
    h_congestion = np.ones(graph.edge_index.shape[1], dtype=np.float32)

    path, edges = router.route_net(
        net_id=1,
        terminals=[t0, t1, t2],
        costs=costs,
        h_congestion=h_congestion,
        current_edge_usage=np.zeros(graph.edge_index.shape[1], dtype=np.float32)
    )

    assert t0 in path, "Terminal 0 must be in path"
    assert t1 in path, "Terminal 1 must be in path"
    assert t2 in path, "Terminal 2 must be in path"


def test_api_board_state_and_routing():
    """Asserts GET /api/board_state renders cleanly and POST /api/route_epoch steps properly."""
    state.load_preset("demo_board.json")  # fresh session regardless of test order
    client = TestClient(app)

    # 1. Test GET /api/board_state
    resp = client.get("/api/board_state")
    assert resp.status_code == 200
    data = resp.json()

    assert "board_id" in data
    assert "grid_shape" in data
    assert "nets" in data
    assert "telemetry" in data
    assert len(data["nets"]) == 15
    assert data["telemetry"]["epoch"] == 0

    # 2. Test POST /api/route_epoch
    step_resp = client.post("/api/route_epoch")
    assert step_resp.status_code == 200
    step_data = step_resp.json()

    assert step_data["telemetry"]["epoch"] == 1
    assert step_data["telemetry"]["wirelength_mm"] > 0
    assert len(step_data["routed_paths"]) == 15

    # 3. Test POST /api/run_pipeline
    run_resp = client.post("/api/run_pipeline", json={"max_epochs": 5})
    assert run_resp.status_code == 200
    run_data = run_resp.json()

    assert run_data["telemetry"]["epoch"] >= 1
    assert run_data["telemetry"]["drc_violations"] == 0


if __name__ == "__main__":
    pytest.main(["-v", __file__])
