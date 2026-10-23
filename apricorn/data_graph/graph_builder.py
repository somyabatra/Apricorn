# FILE: apricorn/data_graph/graph_builder.py
"""
3D Octilinear Grid-Graph Extractor for Project Apricorn.
Member 1: Data & Graph Lead.

Discretizes the board into an L x X x Y GCell mesh with:
- 8 planar compass directions (orthogonal length 1.0, diagonal length 1.4142)
- Vertical via transitions between adjacent layers (l +/- 1, base_cost 4.0)
- Exact GraphDataContract tensor shapes: x_nodes (N, 12), edge_index (2, E), edge_attr (E, 6)
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from apricorn.contracts import GraphDataContract


def _point_in_polygon(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
    """Check if (x, y) is inside or on the boundary of a polygon."""
    try:
        from shapely.geometry import Point, Polygon
        return bool(Polygon(poly).intersects(Point(x, y)))
    except Exception:
        n = len(poly)
        inside = False
        p1x, p1y = poly[0]
        for i in range(n + 1):
            p2x, p2y = poly[i % n]
            if min(p1x, p2x) <= x <= max(p1x, p2x) and min(p1y, p2y) <= y <= max(p1y, p2y):
                if abs((p2y - p1y) * (x - p1x) - (p2x - p1x) * (y - p1y)) < 1e-5:
                    return True
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside


def build_graph(
    board_spec: Dict[str, Any],
    prev_overflow: Optional[np.ndarray] = None,
) -> GraphDataContract:
    """
    Constructs the canonical GraphDataContract from a parsed board specification.

    Args:
        board_spec: Dictionary containing board dimensions, layers, obstacles, and nets.
        prev_overflow: Optional array of shape (E,) containing edge congestion overflow
                       from previous negotiation epochs.

    Returns:
        GraphDataContract conforming to Day-0 mathematical contract.
    """
    board_id = str(board_spec.get("board_id", "unnamed_board"))
    layers = int(board_spec.get("layers", 2))
    width = int(board_spec.get("width", 32))
    height = int(board_spec.get("height", 32))
    grid_shape: Tuple[int, int, int] = (layers, width, height)

    N = layers * width * height

    def node_id(l: int, x: int, y: int) -> int:
        return l * (width * height) + x * height + y

    # 1. Parse Obstacles
    raw_obstacles = board_spec.get("obstacles", [])
    obstacle_polygons: List[Tuple[int, List[Tuple[float, float]]]] = []
    obs_mask = np.zeros(N, dtype=np.float32)

    for obs in raw_obstacles:
        obs_layer = int(obs["layer"])
        pts = [(float(px), float(py)) for px, py in obs["polygon"]]
        obstacle_polygons.append((obs_layer, pts))

        # Discretize obstacle mask onto GCell nodes
        min_x = max(0, int(min(p[0] for p in pts)))
        max_x = min(width - 1, int(max(p[0] for p in pts)))
        min_y = max(0, int(min(p[1] for p in pts)))
        max_y = min(height - 1, int(max(p[1] for p in pts)))

        for gx in range(min_x, max_x + 1):
            for gy in range(min_y, max_y + 1):
                if _point_in_polygon(float(gx), float(gy), pts):
                    obs_mask[node_id(obs_layer, gx, gy)] = 1.0

    # 2. Parse Net Demands and Pin features
    raw_nets = board_spec.get("nets", [])
    total_nets = max(1, len(raw_nets))
    net_demands: Dict[int, List[int]] = {}
    pin_mask = np.zeros(N, dtype=np.float32)
    dem_feat = np.zeros(N, dtype=np.float32)
    net_id_norm = np.full(N, -1.0, dtype=np.float32)

    for net_info in raw_nets:
        nid = int(net_info["net_id"])
        pins = net_info.get("pins", [])
        terminal_nodes: List[int] = []
        for pin in pins:
            pl, px, py = int(pin[0]), int(pin[1]), int(pin[2])
            if 0 <= pl < layers and 0 <= px < width and 0 <= py < height:
                uid = node_id(pl, px, py)
                terminal_nodes.append(uid)
                pin_mask[uid] = 1.0
                dem_feat[uid] += 1.0
                net_id_norm[uid] = float(nid) / float(total_nets)
                obs_mask[uid] = 0.0  # Net terminals override obstacle mask
        if len(terminal_nodes) >= 2:
            net_demands[nid] = terminal_nodes

    # 3. Build Adjacency Graph (8 planar + 2 vertical via edges)
    planar_dirs = [
        (1, 0, 1.0, 1.0),
        (-1, 0, 1.0, 1.0),
        (0, 1, 1.0, 1.0),
        (0, -1, 1.0, 1.0),
        (1, 1, 1.4142, 1.4142),
        (-1, 1, 1.4142, 1.4142),
        (1, -1, 1.4142, 1.4142),
        (-1, -1, 1.4142, 1.4142),
    ]

    src_list: List[int] = []
    dst_list: List[int] = []
    edge_attrs: List[List[float]] = []

    for l in range(layers):
        for x in range(width):
            for y in range(height):
                u = node_id(l, x, y)

                # Planar 8 neighbors
                for dx, dy, length, base_cost in planar_dirs:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        v = node_id(l, nx, ny)
                        src_list.append(u)
                        dst_list.append(v)
                        edge_attrs.append([length, float(dx), float(dy), 0.0, base_cost, 0.0])

                # Vertical via transitions
                for dl in (-1, 1):
                    nl = l + dl
                    if 0 <= nl < layers:
                        v = node_id(nl, x, y)
                        src_list.append(u)
                        dst_list.append(v)
                        edge_attrs.append([1.0, 0.0, 0.0, float(dl), 4.0, 0.0])

    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    edge_attr = np.array(edge_attrs, dtype=np.float32)
    E = edge_index.shape[1]

    # Incorporate prev_overflow into edge_attr and node features if provided
    h_overflow_node = np.zeros(N, dtype=np.float32)
    if prev_overflow is not None and prev_overflow.shape[0] == E:
        overflow_float = prev_overflow.astype(np.float32)
        edge_attr[:, 5] = overflow_float  # feature 5: congestion_ratio
        # Aggregate overflow into incident source nodes
        np.add.at(h_overflow_node, edge_index[0], overflow_float)

    # 4. Populate 12 Node Features:
    # [x, y, z, is_pin, is_obs, cap, dem, h_overflow, net_id_norm, deg, dist_bnd, layer_dir]
    degree_counts = np.bincount(edge_index[0], minlength=N)
    x_nodes = np.zeros((N, 12), dtype=np.float32)

    for l in range(layers):
        for x in range(width):
            for y in range(height):
                u = node_id(l, x, y)
                dist_bnd = float(min(x, width - 1 - x, y, height - 1 - y))
                layer_dir = float(l % 2)  # Even layer: 0 (horiz), Odd layer: 1 (vert)
                x_nodes[u] = [
                    float(x),
                    float(y),
                    float(l),
                    pin_mask[u],
                    obs_mask[u],
                    1.0,  # cap
                    dem_feat[u],
                    h_overflow_node[u],
                    net_id_norm[u],
                    float(degree_counts[u]),
                    dist_bnd,
                    layer_dir,
                ]

    return GraphDataContract(
        board_id=board_id,
        grid_shape=grid_shape,
        x_nodes=x_nodes,
        edge_index=edge_index,
        edge_attr=edge_attr,
        net_demands=net_demands,
        obstacle_polygons=obstacle_polygons,
    )


def update_overflow(
    graph: GraphDataContract,
    edge_overflow: np.ndarray,
) -> GraphDataContract:
    """
    Feedback loop utility connecting Member 3's overflow output back to
    Member 1's graph for epoch t+1.
    """
    E = graph.edge_index.shape[1]
    N = graph.x_nodes.shape[0]

    updated_edge_attr = graph.edge_attr.copy()
    if edge_overflow.shape[0] == E:
        updated_edge_attr[:, 5] = edge_overflow.astype(np.float32)

    updated_x_nodes = graph.x_nodes.copy()
    # Add newly accumulated edge overflow to node h_overflow (feature index 7)
    delta_overflow = np.zeros(N, dtype=np.float32)
    np.add.at(delta_overflow, graph.edge_index[0], edge_overflow.astype(np.float32))
    updated_x_nodes[:, 7] += delta_overflow

    return GraphDataContract(
        board_id=graph.board_id,
        grid_shape=graph.grid_shape,
        x_nodes=updated_x_nodes,
        edge_index=graph.edge_index,
        edge_attr=updated_edge_attr,
        net_demands=graph.net_demands,
        obstacle_polygons=graph.obstacle_polygons,
    )
