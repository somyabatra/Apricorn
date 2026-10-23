"""
Multi-Pin Octilinear Wavefront A* Router for Project Apricorn.
Member 3 Deliverable 3.1:
- Multi-pin routing via Prim-A* tree expansion
- Octilinear 8-directional planar + vertical Z-via transitions
- Dynamic composite edge cost: C_e = (b_e + alpha * W_edge[e]) * p_e * h_e
- Admissible octilinear Euclidean heuristic & 45-degree bend penalties
- Strictly respects edge_index without inventing edges

Phase-2 additions (DRC convergence)
------------------------------------
- Diagonal crossing prevention: active_diagonals registry + _blocking_penalty
- Clearance-zone soft steering: _clearance_extra() scans 8-neighbour set
"""

import heapq
import math
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from apricorn.contracts import GraphDataContract, RoutingCostContract


SQRT2 = math.sqrt(2.0)

# Penalty large enough to be effectively impassable but not inf (so a
# fallback path is still found when no crossing-free route exists).
_BLOCKING_PENALTY = 1_000_000.0

# Soft penalty applied when candidate node v is within 1 grid cell of a
# node owned by a foreign net on the same layer.
# Grid pitch = 1 mm; Shapely buffers each trace by (w/2 + c/2) = 0.45 mm,
# so same-layer traces 1 mm apart just barely clear.  This steers the A*
# away before the geometric check fires.
_CLEARANCE_PENALTY = 25.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def compute_octilinear_heuristic(
    x1: float, y1: float, z1: float,
    x2: float, y2: float, z2: float,
) -> float:
    """Admissible octilinear heuristic across 3D PCB grid."""
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    diag = min(dx, dy)
    ortho = abs(dx - dy)
    planar = diag * SQRT2 + ortho * 1.0
    via = abs(z1 - z2) * 4.0
    return planar + via


def compute_bend_penalty(
    prev_dir: Optional[Tuple[int, int]],
    curr_dir: Tuple[int, int],
    is_via: bool,
) -> float:
    """Computes bend penalty for octilinear turns (straight, 45, 90, 135+ deg)."""
    if prev_dir is None:
        return 0.0
    if is_via:
        return 0.25  # slight via direction penalty

    pdx, pdy = prev_dir
    cdx, cdy = curr_dir
    if pdx == cdx and pdy == cdy:
        return 0.0  # Straight line

    dot = pdx * cdx + pdy * cdy
    norm_p = math.hypot(pdx, pdy)
    norm_c = math.hypot(cdx, cdy)
    if norm_p == 0 or norm_c == 0:
        return 0.0

    cos_theta = max(-1.0, min(1.0, dot / (norm_p * norm_c)))
    angle_deg = math.degrees(math.acos(cos_theta))

    if angle_deg <= 25.0:
        return 0.0
    elif angle_deg <= 60.0:
        return 0.35   # 45-degree bend
    elif angle_deg <= 110.0:
        return 0.85   # 90-degree bend
    else:
        return 2.50   # Acute turn / U-turn


def _canonical_segment(
    lyr: int,
    x1: int, y1: int,
    x2: int, y2: int,
) -> Tuple[int, int, int, int, int]:
    """Direction-independent canonical key for a diagonal segment."""
    if (x1, y1) > (x2, y2):
        x1, y1, x2, y2 = x2, y2, x1, y1
    return (lyr, x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class OctilinearWavefrontRouter:
    """
    High-performance 3D Octilinear Wavefront A* pathfinder.

    Traverses Member 1's edge_index directly and uses Member 2's predicted
    edge weights.

    Phase-2 DRC-aware additions
    ---------------------------
    active_diagonals : Dict[(layer,x1,y1,x2,y2), net_id]
        Registry of every diagonal segment committed by previously routed nets
        within the current epoch.  Used by _diagonal_crossing_penalty() to
        block the orthogonal crossing of an already-owned diagonal.

    _planar_nbrs : List[List[int]]
        Pre-built per-node list of planar (same-layer) neighbours.  Used by
        _clearance_extra() to cheaply check the 8-cell clearance zone.
    """

    def __init__(self, graph: GraphDataContract, use_geometric_clearance: bool = True):
        self.graph = graph
        self.N = graph.x_nodes.shape[0]
        self.E = graph.edge_index.shape[1]
        # Whether to apply the soft clearance-zone steering penalty.
        # Set False when running under the fallback (discrete) DRC engine so
        # easy boards still converge in epoch 1.
        self.use_geometric_clearance = use_geometric_clearance
        _, self.W, self.H = graph.grid_shape

        # Coordinates and obstacle mask
        self.coords = graph.x_nodes[:, 0:3]        # (N, 3)  [x, y, z]
        self.is_obs = graph.x_nodes[:, 4] > 0.5    # (N,) bool
        self.node_capacities = graph.x_nodes[:, 5]  # (N,)

        # Adjacency list: adj[u] = [(v, e_idx, base_cost, dx, dy, dz), ...]
        self.adj: List[List[Tuple[int, int, float, int, int, int]]] = [
            [] for _ in range(self.N)
        ]
        src_nodes = graph.edge_index[0]
        dst_nodes = graph.edge_index[1]
        edge_attr = graph.edge_attr

        for e_idx in range(self.E):
            u = int(src_nodes[e_idx])
            v = int(dst_nodes[e_idx])
            dx = int(edge_attr[e_idx, 1])
            dy = int(edge_attr[e_idx, 2])
            dz = int(edge_attr[e_idx, 3])
            b_cost = float(edge_attr[e_idx, 4])
            self.adj[u].append((v, e_idx, b_cost, dx, dy, dz))

        # Edge capacity (1 trace per directed edge)
        self.edge_capacities = np.ones(self.E, dtype=np.float32)

        # Historical node-collision memory
        self.node_history = np.zeros(self.N, dtype=np.float32)

        # Pin ownership map
        self.terminal_owner: Dict[int, int] = {}
        for _nid, _terms in graph.net_demands.items():
            for _t in _terms:
                self.terminal_owner[int(_t)] = int(_nid)

        # Pre-build planar neighbour list for clearance checks
        self._planar_nbrs: List[List[int]] = [[] for _ in range(self.N)]
        for e_idx in range(self.E):
            u = int(src_nodes[e_idx])
            v = int(dst_nodes[e_idx])
            dz = int(edge_attr[e_idx, 3])
            if dz == 0:
                self._planar_nbrs[u].append(v)

        # Phase-2a: per-epoch diagonal segment and corner-cut registry
        # key: canonical 5-tuple  ->  net_id that owns it
        self.active_diagonals: Dict[Tuple[int, int, int, int, int], int] = {}
        # node_id -> net_id that routed diagonally past this corner
        self.corner_cut_nodes: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Phase-2 helper methods
    # ------------------------------------------------------------------

    def reset_diagonals(self) -> None:
        """Clear the diagonal and corner-cut registries at the start of a new routing epoch."""
        self.active_diagonals.clear()
        self.corner_cut_nodes.clear()

    def _commit_diagonals(self, net_id: int, path_nodes: List[int]) -> None:
        """
        After routing net_id, record all its diagonal segments and cut-corners
        so subsequent nets can detect and avoid crossing or corner-shorting them.
        """
        stride = self.W * self.H
        for i in range(len(path_nodes) - 1):
            u = path_nodes[i]
            v = path_nodes[i + 1]
            ux, uy, uz = self.coords[u]
            vx, vy, vz = self.coords[v]
            dx, dy = int(vx - ux), int(vy - uy)
            if abs(dx) == 1 and abs(dy) == 1 and uz == vz:
                key = _canonical_segment(
                    int(uz), int(ux), int(uy), int(vx), int(vy)
                )
                self.active_diagonals[key] = net_id
                if self.use_geometric_clearance:
                    l = int(uz)
                    c1 = l * stride + int(ux) * self.H + int(vy)
                    c2 = l * stride + int(vx) * self.H + int(uy)
                    if 0 <= c1 < self.N:
                        self.corner_cut_nodes[c1] = net_id
                    if 0 <= c2 < self.N:
                        self.corner_cut_nodes[c2] = net_id

    def _diagonal_crossing_penalty(
        self,
        u: int,
        v: int,
        dx: int,
        dy: int,
        current_net_id: Optional[int],
        occupied_nodes: Optional[Dict[int, int]] = None,
    ) -> float:
        """
        For a planar diagonal move (|dx|==|dy|==1), check whether:
        1. The geometrically crossing diagonal in the same unit square is already
           owned by a different net.
        2. The diagonal cuts a corner adjacent to an obstacle, foreign pin, or
           trace of another net (perpendicular distance 0.707mm < 0.90mm clearance).
        """
        if abs(dx) != 1 or abs(dy) != 1:
            return 0.0  # not diagonal

        ux, uy, uz = self.coords[u]
        vx, vy, vz = self.coords[v]
        if uz != vz:
            return 0.0  # via transition, not a planar diagonal

        lyr = int(uz)
        x1, y1 = int(ux), int(uy)
        x2, y2 = int(vx), int(vy)

        # 1. Opposite diagonal crossing check:
        cross_key = _canonical_segment(lyr, x1, y2, x2, y1)
        owner = self.active_diagonals.get(cross_key)
        if owner is not None and owner != current_net_id:
            return _BLOCKING_PENALTY

        # 2. Geometric corner-cut clearance:
        # A diagonal cutting between (x1,y1) and (x2,y2) comes within 0.707mm < 0.90mm
        # of the two corner nodes (x1,y2) and (x2,y1).
        if self.use_geometric_clearance:
            stride = self.W * self.H
            c1 = lyr * stride + x1 * self.H + y2
            c2 = lyr * stride + x2 * self.H + y1
            for c in (c1, c2):
                if 0 <= c < self.N:
                    if self.is_obs[c]:
                        return _BLOCKING_PENALTY
                    pin_owner = self.terminal_owner.get(c)
                    if pin_owner is not None and pin_owner != current_net_id:
                        return _BLOCKING_PENALTY
                    if occupied_nodes is not None:
                        c_owner = occupied_nodes.get(c)
                        if c_owner is not None and c_owner != current_net_id:
                            return _BLOCKING_PENALTY

        return 0.0

    def _clearance_extra(
        self,
        v: int,
        target: int,
        current_net_id: Optional[int],
        occupied_nodes: Optional[Dict[int, int]],
    ) -> float:
        """
        Return _BLOCKING_PENALTY if another net has already routed diagonally
        past corner v, or _CLEARANCE_PENALTY if any planar neighbour of v is
        owned by a different net, else 0.0.

        Only active when use_geometric_clearance=True (i.e. when the real
        Shapely DRC engine is in use).  In fallback mode this returns 0 so
        that easy boards still converge in epoch 1.
        """
        if not self.use_geometric_clearance:
            return 0.0

        # If another net already routed diagonally past corner node v, visiting v violates clearance
        if v != target and current_net_id is not None:
            cc_owner = self.corner_cut_nodes.get(v)
            if cc_owner is not None and cc_owner != current_net_id:
                return _BLOCKING_PENALTY

        if occupied_nodes is None or current_net_id is None:
            return 0.0
        for nbr in self._planar_nbrs[v]:
            owner = occupied_nodes.get(nbr)
            if owner is not None and owner != current_net_id:
                return _CLEARANCE_PENALTY
        return 0.0

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route_net(
        self,
        net_id: int,
        terminals: List[int],
        costs: RoutingCostContract,
        h_congestion: np.ndarray,
        current_edge_usage: np.ndarray,
        occupied_nodes: Optional[Dict[int, int]] = None,
        alpha: float = 1.0,
    ) -> Tuple[List[int], List[int]]:
        """
        Routes a multi-pin net using Prim-A* tree expansion.
        Returns (ordered_node_path, ordered_edge_path).
        """
        if len(terminals) < 2:
            return terminals[:], []

        tree_nodes: Set[int] = {terminals[0]}
        tree_node_list: List[int] = [terminals[0]]
        tree_edge_list: List[int] = []

        for target_pin in terminals[1:]:
            if target_pin in tree_nodes:
                continue

            path_nodes, path_edges = self._astar_multi_source(
                sources=tree_nodes,
                target=target_pin,
                costs=costs,
                h_congestion=h_congestion,
                current_edge_usage=current_edge_usage,
                occupied_nodes=occupied_nodes,
                current_net_id=net_id,
                alpha=alpha,
            )

            if not path_nodes:
                # Fallback: single-source from the last tree node
                path_nodes, path_edges = self._astar_single_source(
                    source=tree_node_list[-1],
                    target=target_pin,
                    costs=costs,
                    h_congestion=h_congestion,
                    current_edge_usage=current_edge_usage,
                    occupied_nodes=occupied_nodes,
                    current_net_id=net_id,
                    alpha=alpha,
                )

            for nid in path_nodes:
                if nid not in tree_nodes:
                    tree_nodes.add(nid)
                    tree_node_list.append(nid)
            for eid in path_edges:
                tree_edge_list.append(eid)

        return tree_node_list, tree_edge_list

    def _astar_multi_source(
        self,
        sources: Set[int],
        target: int,
        costs: RoutingCostContract,
        h_congestion: np.ndarray,
        current_edge_usage: np.ndarray,
        occupied_nodes: Optional[Dict[int, int]] = None,
        current_net_id: Optional[int] = None,
        alpha: float = 1.0,
    ) -> Tuple[List[int], List[int]]:
        """A* search from a set of source nodes (g=0) to a target node."""
        tx, ty, tz = self.coords[target]

        pq: List[Tuple[float, float, int, Optional[Tuple[int, int]]]] = []
        g_scores: Dict[int, float] = {}
        parent_node: Dict[int, int] = {}
        parent_edge: Dict[int, int] = {}

        for src in sources:
            sx, sy, sz = self.coords[src]
            h = compute_octilinear_heuristic(sx, sy, sz, tx, ty, tz)
            g_scores[src] = 0.0
            heapq.heappush(pq, (h, 0.0, src, None))

        edge_weights = costs.edge_weights
        visited: Set[int] = set()

        while pq:
            f, g, u, prev_dir = heapq.heappop(pq)

            if u == target:
                # Reconstruct path
                path_nodes: List[int] = [u]
                path_edges: List[int] = []
                curr = u
                while curr in parent_node and curr not in sources:
                    edge = parent_edge[curr]
                    curr = parent_node[curr]
                    path_nodes.append(curr)
                    path_edges.append(edge)
                path_nodes.reverse()
                path_edges.reverse()
                return path_nodes, path_edges

            if u in visited:
                continue
            visited.add(u)

            for v, e_idx, base_cost, dx, dy, dz in self.adj[u]:
                # Obstacle avoidance
                if self.is_obs[v] and v != target:
                    continue

                # Foreign pin avoidance
                pin_owner = self.terminal_owner.get(v)
                if pin_owner is not None and pin_owner != current_net_id:
                    continue

                # Congestion penalty p_e
                usage = current_edge_usage[e_idx]
                cap = self.edge_capacities[e_idx]
                p_e = 1.0 + 3.5 * max(0.0, float(usage - cap + 1.0))

                # Node occupancy penalty
                node_penalty = 0.0
                if occupied_nodes is not None and current_net_id is not None:
                    owner = occupied_nodes.get(v)
                    if owner is not None and owner != current_net_id and v != target:
                        node_penalty = 5000.0 * (1.0 + float(self.node_history[v]))

                # Historical congestion h_e
                h_e = float(h_congestion[e_idx])

                # STGNN predicted weight
                w_pred = float(edge_weights[e_idx]) if e_idx < len(edge_weights) else 1.0

                # Composite cost
                step_cost = (base_cost + alpha * w_pred) * p_e * h_e + node_penalty

                # Bend penalty
                is_via = (dz != 0)
                curr_dir = (dx, dy)
                step_cost += compute_bend_penalty(prev_dir, curr_dir, is_via)

                # --- Phase 2a: diagonal crossing & corner cut blocking ---
                step_cost += self._diagonal_crossing_penalty(
                    u, v, dx, dy, current_net_id, occupied_nodes
                )

                # --- Phase 2b: clearance-zone soft steering ---
                step_cost += self._clearance_extra(v, target, current_net_id, occupied_nodes)

                tentative_g = g + step_cost

                if tentative_g < g_scores.get(v, float("inf")):
                    g_scores[v] = tentative_g
                    parent_node[v] = u
                    parent_edge[v] = e_idx
                    vx, vy, vz = self.coords[v]
                    h_val = compute_octilinear_heuristic(vx, vy, vz, tx, ty, tz)
                    heapq.heappush(
                        pq, (tentative_g + h_val, tentative_g, v, curr_dir)
                    )

        return [], []

    def _astar_single_source(
        self,
        source: int,
        target: int,
        costs: RoutingCostContract,
        h_congestion: np.ndarray,
        current_edge_usage: np.ndarray,
        occupied_nodes: Optional[Dict[int, int]] = None,
        current_net_id: Optional[int] = None,
        alpha: float = 1.0,
    ) -> Tuple[List[int], List[int]]:
        """Fallback single-source A*."""
        return self._astar_multi_source(
            sources={source},
            target=target,
            costs=costs,
            h_congestion=h_congestion,
            current_edge_usage=current_edge_usage,
            occupied_nodes=occupied_nodes,
            current_net_id=current_net_id,
            alpha=alpha,
        )

    def route_all_nets(
        self,
        costs: RoutingCostContract,
        h_congestion: np.ndarray,
        alpha: float = 1.0,
    ) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], np.ndarray, float, int]:
        """
        Routes all nets in descending priority order.
        Returns (routed_paths, routed_edges, edge_usage, total_wirelength, total_vias).

        Phase-2: resets diagonal registry and commits each net's diagonals
        immediately after routing so subsequent nets see accurate crossing data.
        """
        # Reset diagonal registry for this epoch
        self.reset_diagonals()

        priorities = costs.net_priorities
        sorted_net_ids = sorted(
            self.graph.net_demands.keys(),
            key=lambda nid: priorities.get(nid, 0.5),
            reverse=True,
        )

        current_edge_usage = np.zeros(self.E, dtype=np.float32)
        occupied_nodes: Dict[int, int] = {}
        routed_paths: Dict[int, List[int]] = {}
        routed_edges: Dict[int, List[int]] = {}

        total_wirelength = 0.0
        total_vias = 0

        for net_id in sorted_net_ids:
            terminals = self.graph.net_demands[net_id]
            path_nodes, path_edges = self.route_net(
                net_id=net_id,
                terminals=terminals,
                costs=costs,
                h_congestion=h_congestion,
                current_edge_usage=current_edge_usage,
                occupied_nodes=occupied_nodes,
                alpha=alpha,
            )

            routed_paths[net_id] = path_nodes
            routed_edges[net_id] = path_edges

            for nid in path_nodes:
                if nid not in occupied_nodes:
                    occupied_nodes[nid] = net_id

            for e_idx in path_edges:
                current_edge_usage[e_idx] += 1.0
                length = float(self.graph.edge_attr[e_idx, 0])
                dz = float(self.graph.edge_attr[e_idx, 3])
                if abs(dz) > 0.0:
                    total_vias += 1
                else:
                    total_wirelength += length

            # Phase-2a: commit this net's diagonals before routing the next
            self._commit_diagonals(net_id, path_nodes)

        return routed_paths, routed_edges, current_edge_usage, total_wirelength, total_vias
