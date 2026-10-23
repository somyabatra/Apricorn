# FILE: apricorn/stgnn_engine/labels.py
"""
Training labels + router rollouts for the STGNN (numpy only; no torch needed).

Two supervision sources:

1. IMITATION  - every synthetic board from Member 1 carries `ground_truth_path`
   for each net (a known-good solution). `gt_edge_mask` marks the directed edges
   those paths use. The model should give these edges LOW weight.

2. CONGESTION - `collect_rollout` runs the REAL negotiated router for T epochs
   with neutral costs and records, per epoch, which edges conflicted and which
   nets had to be ripped up. The model should give conflicting edges HIGH weight
   and route frequently-ripped-up (hard) nets FIRST (high priority).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from apricorn.contracts import GraphDataContract, RoutingCostContract
from apricorn.stgnn_engine.feedback import conflict_signal


def uniform_costs(graph: GraphDataContract, epoch: int = 1) -> RoutingCostContract:
    """Neutral 'no AI' baseline: every edge weight 1.0, every net priority 0.5."""
    E = graph.edge_index.shape[1]
    N = graph.x_nodes.shape[0]
    return RoutingCostContract(
        epoch=epoch,
        edge_weights=np.ones(E, dtype=np.float32),
        net_priorities={nid: 0.5 for nid in graph.net_demands},
        hidden_state=np.zeros((N, 64), dtype=np.float32),
    )


class UniformPredictor:
    """Drop-in for STGNNPredictor.predict(): the non-AI baseline."""
    name = "uniform"

    def predict(self, graph, prev_hidden, epoch):
        return uniform_costs(graph, epoch)


def _node_id(spec: Dict[str, Any], l: int, x: int, y: int) -> int:
    W, H = int(spec["width"]), int(spec["height"])
    return l * (W * H) + x * H + y


def gt_edge_mask(graph: GraphDataContract, board_spec: Dict[str, Any]) -> np.ndarray:
    """(E,) bool: directed edges (both directions) used by the ground-truth routes."""
    src, dst = graph.edge_index
    lookup = {(int(u), int(v)): i for i, (u, v) in enumerate(zip(src, dst))}
    mask = np.zeros(src.shape[0], dtype=bool)
    for net in board_spec.get("nets", []):
        path = net.get("ground_truth_path")
        if not path:
            continue
        ids = [_node_id(board_spec, int(c[0]), int(c[1]), int(c[2])) for c in path]
        for u, v in zip(ids[:-1], ids[1:]):
            for key in ((u, v), (v, u)):
                i = lookup.get(key)
                if i is not None:
                    mask[i] = True
    return mask


@dataclass
class Rollout:
    """Router behaviour on one board under neutral costs, T epochs."""
    conflict_seq: List[np.ndarray] = field(default_factory=list)   # T x (E,) float32
    drc_seq: List[int] = field(default_factory=list)
    ripup: Dict[int, int] = field(default_factory=dict)            # net_id -> times ripped up
    converged_epoch: Optional[int] = None


def collect_rollout(
    graph: GraphDataContract,
    T: int = 6,
    drc_fn: Optional[Callable] = None,
) -> Rollout:
    """Run the real router for up to T epochs with neutral costs; record conflicts."""
    from apricorn.data_graph.graph_builder import update_overflow
    from apricorn.router_ui.negotiator import NegotiatedRouter

    router = NegotiatedRouter(drc_fn=drc_fn, max_epochs=T)
    ro = Rollout()
    g = graph
    for t in range(1, T + 1):
        res = router.step_epoch(g, uniform_costs(g, t), epoch=t)
        sig = conflict_signal(g, res)
        ro.conflict_seq.append(sig)
        ro.drc_seq.append(int(res.drc_violations))
        if res.drc_violations == 0 and float(res.edge_overflow.sum()) == 0.0:
            ro.converged_epoch = t
            break
        g = update_overflow(g, sig)
    ro.ripup = dict(router.ripup_counts)
    return ro


def priority_targets(graph: GraphDataContract, ro: Rollout) -> Dict[int, float]:
    """
    Net-priority labels in [0.1, 0.9]: nets that were ripped up most often are the
    hard ones and should be routed first. Returns {} when the rollout had no
    conflicts (no information, so no priority loss for that board).
    """
    if not ro.ripup or max(ro.ripup.values()) == 0:
        return {}
    m = float(max(ro.ripup.values()))
    return {nid: 0.1 + 0.8 * (ro.ripup.get(nid, 0) / m) for nid in graph.net_demands}


def graph_sequence(graph: GraphDataContract, ro: Rollout) -> List[GraphDataContract]:
    """
    The graphs the model should SEE at each step: step 0 has no congestion,
    step t has the conflicts from epochs < t fed back (exactly as at inference).
    """
    from apricorn.data_graph.graph_builder import update_overflow

    seq = [graph]
    for sig in ro.conflict_seq[:-1]:
        seq.append(update_overflow(seq[-1], sig))
    return seq
