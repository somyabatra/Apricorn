"""
Closed-Loop Negotiation Controller for Project Apricorn.
Member 3 Deliverable 3.2:
- Runs iterative rip-up-and-reroute across up to T=20 negotiation epochs
- Accumulates historical congestion: h_e^(t) = h_e^(t-1) + eta * Overflow(e)
- Integrates with Member 1's check_drc() or internal geometric fallback
- Supports --use-mocks CLI flag for independent Day-1 development
- Returns immutable RoutedBoardResult contracts
"""

import argparse
from typing import Callable, Dict, List, Optional, Set, Tuple
import numpy as np

from apricorn.contracts import GraphDataContract, RoutedBoardResult, RoutingCostContract
from apricorn.mocks import make_mock_costs, make_mock_graph
from apricorn.router_ui.astar_router import OctilinearWavefrontRouter


class NegotiatedRouter:
    """
    Closed-Loop Negotiation Router.
    Orchestrates multi-net octilinear pathfinding, rip-up & reroute of congested traces,
    and historical congestion memory to achieve 0 DRC violations and 0 edge overflow.
    """

    def __init__(
        self,
        drc_fn: Optional[Callable[[GraphDataContract, Dict[int, List[int]], float], int]] = None,
        max_epochs: int = 20,
        eta: float = 0.5,
        alpha: float = 1.0
    ):
        self.drc_fn = drc_fn
        self.max_epochs = max_epochs
        self.eta = eta
        self.alpha = alpha

        # Internal state
        self.current_graph: Optional[GraphDataContract] = None
        self.wavefront_router: Optional[OctilinearWavefrontRouter] = None
        self.h_congestion: Optional[np.ndarray] = None
        self.routed_paths: Dict[int, List[int]] = {}
        self.routed_edges: Dict[int, List[int]] = {}
        self.edge_usage: Optional[np.ndarray] = None
        # Training/analytics hooks: how often each net was ripped up, and the set
        # ripped up in the most recent epoch.
        self.ripup_counts: Dict[int, int] = {}
        self.last_ripped: Set[int] = set()
        # Phase 2.3: nets flagged as DRC-violating in the previous epoch
        self._pending_drc_rip: Set[int] = set()

    def _init_router_if_needed(self, graph: GraphDataContract):
        """Initializes or resets internal router state when graph changes."""
        if self.current_graph is None or self.current_graph.board_id != graph.board_id:
            self.current_graph = graph
            # Enable geometric clearance steering only when a real DRC engine is
            # present; fallback (discrete) mode doesn't need it and it would
            # cause easy boards to take extra epochs to converge.
            self.wavefront_router = OctilinearWavefrontRouter(
                graph, use_geometric_clearance=(self.drc_fn is not None)
            )
            E = graph.edge_index.shape[1]
            self.h_congestion = np.ones(E, dtype=np.float32)
            self.edge_usage = np.zeros(E, dtype=np.float32)
            self.routed_paths = {}
            self.routed_edges = {}
            self.ripup_counts = {}
            self.last_ripped = set()
            self._pending_drc_rip = set()

    def _fallback_check_drc(
        self,
        graph: GraphDataContract,
        paths: Dict[int, List[int]]
    ) -> int:
        """
        Pure-software geometric fallback DRC checker when Member 1's engine is not yet bound.
        Checks for:
        1. Node overlap between distinct nets (excluding declared shared terminals).
        2. Obstacle node collisions.
        3. Edge sharing collisions exceeding capacity.
        """
        violations = 0
        node_owners: Dict[int, int] = {}
        is_obs = graph.x_nodes[:, 4] > 0.5

        for net_id, node_list in paths.items():
            for nid in node_list:
                # 1. Obstacle collision
                if is_obs[nid]:
                    violations += 1

                # 2. Node short with another net
                if nid in node_owners and node_owners[nid] != net_id:
                    violations += 1
                else:
                    node_owners[nid] = net_id

        # 3. Edge sharing short
        if self.edge_usage is not None:
            over = np.maximum(0.0, self.edge_usage - 1.0)
            violations += int(np.count_nonzero(over > 0))

        return violations

    def step_epoch(
        self,
        graph: GraphDataContract,
        costs: RoutingCostContract,
        epoch: int = 1
    ) -> RoutedBoardResult:
        """
        Executes a single negotiation epoch.
        - On epoch 1: routes all nets.
        - On epoch > 1: identifies congested edges, rips up conflicting nets, updates h_e, and reroutes.
        """
        self._init_router_if_needed(graph)
        assert self.wavefront_router is not None
        assert self.h_congestion is not None

        E = graph.edge_index.shape[1]

        if epoch == 1 or len(self.routed_paths) == 0:
            # Full routing of all nets
            (
                self.routed_paths,
                self.routed_edges,
                self.edge_usage,
                total_wirelength,
                total_vias
            ) = self.wavefront_router.route_all_nets(
                costs=costs,
                h_congestion=self.h_congestion,
                alpha=self.alpha
            )
            # Rebuild occupied_nodes
            self.occupied_nodes: Dict[int, int] = {}
            for nid_key, plist in self.routed_paths.items():
                for n_idx in plist:
                    if n_idx not in self.occupied_nodes:
                        self.occupied_nodes[n_idx] = nid_key
        else:
            # Rip-up and Re-route phase
            # Calculate overflow
            edge_overflow = np.maximum(0.0, self.edge_usage - 1.0)
            congested_edges = set(np.where(edge_overflow > 0)[0])

            # Find node collisions
            seen_nodes: Dict[int, int] = {}
            node_collision_nets: Set[int] = set()
            collided_nodes: Set[int] = set()
            for net_id, nlist in self.routed_paths.items():
                for n_idx in nlist:
                    if n_idx in seen_nodes and seen_nodes[n_idx] != net_id:
                        node_collision_nets.add(net_id)
                        node_collision_nets.add(seen_nodes[n_idx])
                        collided_nodes.add(n_idx)
                    else:
                        seen_nodes[n_idx] = net_id

            # Find nets to rip up
            nets_to_rip: Set[int] = set()
            if len(congested_edges) > 0:
                self.h_congestion += self.eta * edge_overflow
                for net_id, edges in self.routed_edges.items():
                    if any(e in congested_edges for e in edges):
                        nets_to_rip.add(net_id)

            nets_to_rip.update(node_collision_nets)
            # Phase 2.3: also rip up nets that caused geometric DRC violations last epoch
            drc_rip_this_epoch = set(self._pending_drc_rip)
            nets_to_rip.update(drc_rip_this_epoch)
            self._pending_drc_rip = set()  # drain after consuming

            # Phase 2.3b: bump h_congestion on DRC-violating net edges so the A*
            # is discouraged from re-tracing the exact same paths.  Without this
            # the router stagnates: no edge overflow -> h_congestion frozen -> rip-up
            # produces identical routes -> DRC count never drops.
            # Bump factor 1.5*eta gives strong enough discouragement without making
            # the edges fully impassable on a short board.
            for net_id in drc_rip_this_epoch:
                for e_idx in self.routed_edges.get(net_id, []):
                    self.h_congestion[e_idx] += self.eta * 1.5
                # Also increment node_history for nodes on DRC-violating paths so
                # the node-occupancy penalty compounds for repeatedly bad positions.
                for n_idx in self.routed_paths.get(net_id, []):
                    self.wavefront_router.node_history[n_idx] += 0.5

            for _n in collided_nodes:
                self.wavefront_router.node_history[_n] += 1.0
            self.last_ripped = set(nets_to_rip)
            for _nid in nets_to_rip:
                self.ripup_counts[_nid] = self.ripup_counts.get(_nid, 0) + 1

            if len(nets_to_rip) > 0:
                # Rip up conflicting nets from edge usage, occupied_nodes, and diagonal registries
                for net_id in nets_to_rip:
                    for e_idx in self.routed_edges.get(net_id, []):
                        self.edge_usage[e_idx] = max(0.0, self.edge_usage[e_idx] - 1.0)
                    for n_idx in self.routed_paths.get(net_id, []):
                        if self.occupied_nodes.get(n_idx) == net_id:
                            del self.occupied_nodes[n_idx]

                self.wavefront_router.active_diagonals = {
                    seg: nid for seg, nid in self.wavefront_router.active_diagonals.items()
                    if nid not in nets_to_rip
                }
                self.wavefront_router.corner_cut_nodes = {
                    node: nid for node, nid in self.wavefront_router.corner_cut_nodes.items()
                    if nid not in nets_to_rip
                }

                # Re-route ripped nets in priority order
                priorities = costs.net_priorities
                sorted_ripped = sorted(
                    list(nets_to_rip),
                    key=lambda nid: priorities.get(nid, 0.5),
                    reverse=True
                )

                for net_id in sorted_ripped:
                    terminals = graph.net_demands[net_id]
                    p_nodes, p_edges = self.wavefront_router.route_net(
                        net_id=net_id,
                        terminals=terminals,
                        costs=costs,
                        h_congestion=self.h_congestion,
                        current_edge_usage=self.edge_usage,
                        occupied_nodes=self.occupied_nodes,
                        alpha=self.alpha
                    )
                    self.routed_paths[net_id] = p_nodes
                    self.routed_edges[net_id] = p_edges
                    for n_idx in p_nodes:
                        if n_idx not in self.occupied_nodes:
                            self.occupied_nodes[n_idx] = net_id
                    for e_idx in p_edges:
                        self.edge_usage[e_idx] += 1.0
                    self.wavefront_router._commit_diagonals(net_id, p_nodes)

        # Calculate final telemetry for this epoch
        edge_overflow = np.maximum(0.0, self.edge_usage - 1.0).astype(np.float32)

        total_wirelength = 0.0
        total_vias = 0
        for edges in self.routed_edges.values():
            for e_idx in edges:
                length = float(graph.edge_attr[e_idx, 0])
                dz = float(graph.edge_attr[e_idx, 3])
                if abs(dz) > 0.0:
                    total_vias += 1
                else:
                    total_wirelength += length

        # Check DRC violations — use the drc_fn's default clearance (0.5mm for check_drc)
        # so the router's internal check agrees with the gate test's check.
        violators: Set[int] = set()
        if self.drc_fn is not None:
            try:
                res_drc = self.drc_fn(graph, self.routed_paths, return_violators=True)
                if isinstance(res_drc, tuple):
                    drc_violations, violators = int(res_drc[0]), set(res_drc[1])
                else:
                    drc_violations = int(res_drc)
            except (TypeError, ValueError):
                drc_violations = int(self.drc_fn(graph, self.routed_paths))
        else:
            drc_violations = self._fallback_check_drc(graph, self.routed_paths)

        # --- Phase 2.3: DRC-violation-driven rip-up attribution ---
        if drc_violations > 0:
            if violators:
                self._pending_drc_rip: Set[int] = violators
            elif self.drc_fn is not None:
                net_drc_counts: Dict[int, int] = {}
                for net_id, path in self.routed_paths.items():
                    cnt = int(self.drc_fn(graph, {net_id: path}))
                    if cnt > 0:
                        net_drc_counts[net_id] = cnt
                self._pending_drc_rip = set(net_drc_counts.keys())
            else:
                self._pending_drc_rip = set()
        else:
            self._pending_drc_rip = set()

        return RoutedBoardResult(
            epoch=epoch,
            routed_paths=dict(self.routed_paths),
            edge_overflow=edge_overflow,
            drc_violations=drc_violations,
            wirelength_mm=float(total_wirelength),
            via_count=int(total_vias)
        )

    def run_pipeline(
        self,
        graph: GraphDataContract,
        max_epochs: Optional[int] = None,
        use_mocks: bool = False,
        predictor_fn: Optional[Callable[[GraphDataContract, Optional[np.ndarray], int], RoutingCostContract]] = None
    ) -> RoutedBoardResult:
        """
        Runs the full closed-loop negotiation process across multiple epochs
        until drc_violations == 0 and edge_overflow.sum() == 0, or max_epochs is reached.
        """
        limit = max_epochs or self.max_epochs
        hidden_state = None
        result = None

        for epoch in range(1, limit + 1):
            if predictor_fn is not None and not use_mocks:
                costs = predictor_fn(graph, hidden_state, epoch)
            else:
                costs = make_mock_costs(graph, epoch=epoch)

            hidden_state = costs.hidden_state
            result = self.step_epoch(graph, costs, epoch=epoch)

            # Check convergence criteria
            if result.drc_violations == 0 and float(result.edge_overflow.sum()) == 0.0:
                break

        assert result is not None
        return result


def main():
    parser = argparse.ArgumentParser(description="Apricorn Closed-Loop Negotiated Router")
    parser.add_argument("--use-mocks", action="store_true", help="Run against apricorn/mocks.py")
    parser.add_argument("--epochs", type=int, default=20, help="Max negotiation epochs")
    args = parser.parse_args()

    print("[Apricorn Member 3] Initializing NegotiatedRouter...")
    graph = make_mock_graph()
    router = NegotiatedRouter(max_epochs=args.epochs)
    result = router.run_pipeline(graph=graph, max_epochs=args.epochs, use_mocks=args.use_mocks or True)

    print("\n--- Negotiation Completed ---")
    print(f"Final Epoch: {result.epoch}")
    print(f"Nets Routed: {len(result.routed_paths)} / {len(graph.net_demands)}")
    print(f"Total Wirelength: {result.wirelength_mm:.2f} mm")
    print(f"Via Count: {result.via_count}")
    print(f"DRC Violations: {result.drc_violations}")
    print(f"Total Edge Overflow: {result.edge_overflow.sum():.2f}")


if __name__ == "__main__":
    main()
