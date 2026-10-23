"""Numpy-only tests for the training-signal plumbing (no torch required)."""
import numpy as np

from apricorn.contracts import RoutedBoardResult
from apricorn.data_graph.graph_builder import build_graph
from apricorn.data_graph.parser_gen import generate_synthetic_board
from apricorn.stgnn_engine.feedback import collided_nodes, conflict_signal
from apricorn.stgnn_engine.labels import (
    collect_rollout, graph_sequence, gt_edge_mask, priority_targets, uniform_costs,
)


def _dense():
    spec = generate_synthetic_board(2, 20, 20, 40, seed=2)
    return spec, build_graph(spec)


def test_gt_edge_mask_is_valid_and_symmetric():
    spec, g = _dense()
    m = gt_edge_mask(g, spec)
    assert m.shape == (g.edge_index.shape[1],) and m.dtype == bool
    assert 0 < m.sum() < m.size
    src, dst = g.edge_index
    fwd = {(int(u), int(v)) for u, v in zip(src[m], dst[m])}
    assert all((v, u) in fwd for u, v in fwd), "ground-truth mask must cover both directions"


def test_collided_nodes_and_conflict_signal():
    spec, g = _dense()
    E = g.edge_index.shape[1]
    res = RoutedBoardResult(epoch=1, routed_paths={0: [1, 2, 3], 1: [3, 4], 2: [9, 10]},
                            edge_overflow=np.zeros(E, np.float32), drc_violations=1,
                            wirelength_mm=1.0, via_count=0)
    assert collided_nodes(res.routed_paths) == {3}
    sig = conflict_signal(g, res)
    src, dst = g.edge_index
    touching = (src == 3) | (dst == 3)
    assert sig.shape == (E,) and sig.dtype == np.float32
    assert np.all(sig[touching] >= 1.0) and np.all(sig[~touching] == 0.0)


def test_rollout_records_conflicts_and_feeds_them_back():
    spec, g = _dense()
    ro = collect_rollout(g, T=4)
    assert 1 <= len(ro.conflict_seq) <= 4 and len(ro.drc_seq) == len(ro.conflict_seq)
    assert ro.conflict_seq[0].sum() > 0, "dense board should produce conflicts under neutral costs"
    seq = graph_sequence(g, ro)
    assert len(seq) == len(ro.conflict_seq)
    assert seq[0].x_nodes[:, 7].max() == 0.0                       # step 0 sees no congestion
    if len(seq) > 1:
        assert seq[1].x_nodes[:, 7].max() > 0.0                    # step 1 sees step-0 conflicts
    pt = priority_targets(g, ro)
    assert pt and all(0.1 <= v <= 0.9 for v in pt.values())


def test_easy_board_gives_no_priority_signal():
    g = build_graph(generate_synthetic_board(2, 32, 32, 15, seed=1000))
    ro = collect_rollout(g, T=4)
    assert ro.converged_epoch == 1 and priority_targets(g, ro) == {}


def test_uniform_costs_obey_contract():
    _, g = _dense()
    c = uniform_costs(g)
    assert c.edge_weights.shape == (g.edge_index.shape[1],) and np.all(c.edge_weights >= 1.0)
    assert c.hidden_state.shape == (g.x_nodes.shape[0], 64)
