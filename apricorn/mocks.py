# FILE: apricorn/mocks.py
# MERGED: Best-of-both from Member 1 and Member 2 implementations.
#
# Key differences resolved:
#   - make_mock_graph: uses M1's full 8-direction + via adjacency (more realistic)
#     and M1's proper pin flagging (is_pin=1.0 for terminals, guaranteed by gate test)
#   - make_mock_costs: uses M1's flexible signature (accepts GraphDataContract or int)
#     combined with M2's epoch-seeded RNG variant for reproducibility
#
# Both Member 2 (stgnn_engine) and Member 3 (router_ui) import from here on Day-1.

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from apricorn.contracts import GraphDataContract, RoutingCostContract


def make_mock_graph(
    L: int = 2,
    X: int = 32,
    Y: int = 32,
    nets: int = 15,
    seed: int = 42,
) -> GraphDataContract:
    """
    Generate a deterministic synthetic 3D octilinear grid-graph conforming strictly
    to GraphDataContract — used by M2 and M3 for Day-1 parallel development.

    Uses full 8-planar-direction + via adjacency (same as M1's real graph_builder).
    Terminal nodes are guaranteed to have is_pin=1.0 (required by M1 gate test).

    Args:
        L:    Number of PCB layers.
        X:    Grid width.
        Y:    Grid height.
        nets: Number of nets; each gets exactly 2 terminal pins.
        seed: RNG seed for full reproducibility across all members.

    Returns:
        A frozen GraphDataContract matching all contract shapes exactly.
    """
    rng = np.random.default_rng(seed)
    N = L * X * Y

    def node_id(l: int, x: int, y: int) -> int:
        return l * (X * Y) + x * Y + y

    # ---- Build edges: 8 planar compass directions + vertical vias ---------
    # Matches M1's graph_builder.py exactly for consistency
    planar_dirs = [
        (1,  0,  1.0,    1.0),     # right (orthogonal)
        (-1, 0,  1.0,    1.0),     # left
        (0,  1,  1.0,    1.0),     # up
        (0,  -1, 1.0,    1.0),     # down
        (1,  1,  1.4142, 1.4142),  # diagonal NE
        (-1, 1,  1.4142, 1.4142),  # diagonal NW
        (1,  -1, 1.4142, 1.4142),  # diagonal SE
        (-1, -1, 1.4142, 1.4142),  # diagonal SW
    ]

    src_list: List[int] = []
    dst_list: List[int] = []
    edge_attrs: List[List[float]] = []

    for l in range(L):
        for x in range(X):
            for y in range(Y):
                u = node_id(l, x, y)

                # Planar 8 neighbors
                for dx, dy, length, base_cost in planar_dirs:
                    nx2, ny2 = x + dx, y + dy
                    if 0 <= nx2 < X and 0 <= ny2 < Y:
                        v = node_id(l, nx2, ny2)
                        src_list.append(u)
                        dst_list.append(v)
                        edge_attrs.append([length, float(dx), float(dy), 0.0, base_cost, 0.0])

                # Vertical via transitions (both directions for bidirectionality)
                for dl in (-1, 1):
                    nl = l + dl
                    if 0 <= nl < L:
                        v = node_id(nl, x, y)
                        src_list.append(u)
                        dst_list.append(v)
                        edge_attrs.append([1.0, 0.0, 0.0, float(dl), 4.0, 0.0])

    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    edge_attr = np.array(edge_attrs, dtype=np.float32)
    E = edge_index.shape[1]

    # ---- Net demands: 2 non-overlapping terminal pins per net -------------
    all_nodes = np.arange(N)
    chosen_terminals = rng.choice(all_nodes, size=min(nets * 2, N), replace=False)
    net_demands: Dict[int, List[int]] = {}
    pin_mask = np.zeros(N, dtype=np.float32)
    net_id_norm = np.full(N, -1.0, dtype=np.float32)

    for i in range(nets):
        t1 = int(chosen_terminals[2 * i])
        t2 = int(chosen_terminals[2 * i + 1])
        net_demands[i] = [t1, t2]
        pin_mask[t1] = 1.0
        pin_mask[t2] = 1.0
        net_id_norm[t1] = float(i) / max(1.0, float(nets))
        net_id_norm[t2] = float(i) / max(1.0, float(nets))

    # ---- Obstacle regions -------------------------------------------------
    obstacle_polygons: List[Tuple[int, List[Tuple[float, float]]]] = [
        (0, [(10.0, 10.0), (14.0, 10.0), (14.0, 14.0), (10.0, 14.0)]),
        (1, [(18.0, 18.0), (22.0, 18.0), (22.0, 22.0), (18.0, 22.0)]),
    ]
    obs_mask = np.zeros(N, dtype=np.float32)
    for l in range(L):
        for x in range(X):
            for y in range(Y):
                u = node_id(l, x, y)
                if pin_mask[u] == 1.0:
                    continue  # pins always override obstacles
                if l == 0 and 10 <= x <= 14 and 10 <= y <= 14:
                    obs_mask[u] = 1.0
                elif l == 1 and 18 <= x <= 22 and 18 <= y <= 22:
                    obs_mask[u] = 1.0

    # ---- Node features (N, 12) --------------------------------------------
    degree_counts = np.bincount(edge_index[0], minlength=N)
    x_nodes = np.zeros((N, 12), dtype=np.float32)

    for l in range(L):
        for x in range(X):
            for y in range(Y):
                u = node_id(l, x, y)
                dist_bnd = float(min(x, X - 1 - x, y, Y - 1 - y))
                layer_dir = float(l % 2)  # 0=horizontal on even layers, 1=vertical on odd
                x_nodes[u] = [
                    float(x),
                    float(y),
                    float(l),
                    pin_mask[u],
                    obs_mask[u],
                    1.0,                                       # cap
                    1.0 if pin_mask[u] > 0 else 0.0,          # dem
                    0.0,                                       # h_overflow (epoch 0)
                    net_id_norm[u],
                    float(degree_counts[u]),
                    dist_bnd,
                    layer_dir,
                ]

    return GraphDataContract(
        board_id=f"mock_board_{L}x{X}x{Y}_seed{seed}",
        grid_shape=(L, X, Y),
        x_nodes=x_nodes,
        edge_index=edge_index,
        edge_attr=edge_attr,
        net_demands=net_demands,
        obstacle_polygons=obstacle_polygons,
    )


def make_mock_costs(
    E: Union[int, "GraphDataContract"] = 15000,
    N: Optional[int] = None,
    nets: int = 15,
    epoch: int = 1,
    seed: int = 42,
) -> RoutingCostContract:
    """
    Generate a deterministic synthetic RoutingCostContract for Member 3's development.

    Accepts either a raw edge count or a GraphDataContract (M1's flexible signature).
    RNG is seeded with seed+epoch so each negotiation epoch produces distinct mocks.

    Args:
        E:     Number of directed edges, OR a GraphDataContract to extract E/N/nets from.
        N:     Number of nodes (ignored if E is a GraphDataContract).
        nets:  Number of nets (ignored if E is a GraphDataContract).
        epoch: Current negotiation epoch — offsets the RNG seed for variation.
        seed:  Base RNG seed.

    Returns:
        A frozen RoutingCostContract with all shapes/constraints satisfied.
    """
    rng = np.random.default_rng(seed + epoch)

    if isinstance(E, GraphDataContract):
        edge_count = E.edge_index.shape[1]
        node_count = E.x_nodes.shape[0]
        net_keys = list(E.net_demands.keys())
    else:
        edge_count = int(E)
        node_count = N if N is not None else 2048
        net_keys = list(range(nets))

    # edge_weights: strictly >= 1.0 (mirrors 1+softplus guarantee from heads.py)
    edge_weights = (1.0 + rng.uniform(0.0, 2.5, size=edge_count)).astype(np.float32)

    # net_priorities: in (0.0, 1.0) open interval
    net_priorities: Dict[int, float] = {
        int(k): float(rng.uniform(0.1, 0.99)) for k in net_keys
    }

    # hidden_state: (N, 64) temporal GRU embedding
    hidden_state = rng.standard_normal((node_count, 64)).astype(np.float32)

    return RoutingCostContract(
        epoch=epoch,
        edge_weights=edge_weights,
        net_priorities=net_priorities,
        hidden_state=hidden_state,
    )
