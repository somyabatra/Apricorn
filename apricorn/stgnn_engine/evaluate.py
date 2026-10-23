# FILE: apricorn/stgnn_engine/evaluate.py
"""
Does the STGNN actually help? Route the SAME held-out boards with the SAME router and
the SAME closed feedback loop, changing only where the edge costs / net order come from:

    uniform          neutral costs (no AI)                       <- the baseline to beat
    random-mock      apricorn.mocks.make_mock_costs (random)     <- sanity check
    stgnn-untrained  the network with random weights
    stgnn-trained    the network with checkpoints/stgnn_v1.pt

    python -m apricorn.stgnn_engine.evaluate
    python -m apricorn.stgnn_engine.evaluate --drc fallback    # if shapely is unavailable

Writes checkpoints/eval_report.json (served at /api/eval_report).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Callable, Dict, List, Optional

import numpy as np

from apricorn.stgnn_engine.dataset import generate_mixed_dataset, list_boards, load_spec
from apricorn.stgnn_engine.feedback import conflict_signal
from apricorn.stgnn_engine.labels import UniformPredictor


class MockPredictor:
    name = "random-mock"

    def predict(self, graph, prev_hidden, epoch):
        from apricorn.mocks import make_mock_costs
        return make_mock_costs(graph, epoch=epoch)


def run_closed_loop(graph, predictor, drc_fn: Optional[Callable], max_epochs: int = 15) -> Dict:
    """The exact loop used by main.py / the server: predict -> route -> feedback."""
    from apricorn.data_graph.graph_builder import update_overflow
    from apricorn.router_ui.negotiator import NegotiatedRouter

    router = NegotiatedRouter(drc_fn=drc_fn, max_epochs=max_epochs)
    hidden = None
    t0 = time.time()
    res = None
    converged_at = None
    for epoch in range(1, max_epochs + 1):
        costs = predictor.predict(graph, hidden, epoch)
        hidden = costs.hidden_state
        res = router.step_epoch(graph, costs, epoch=epoch)
        if res.drc_violations == 0 and float(res.edge_overflow.sum()) == 0.0:
            converged_at = epoch
            break
        graph = update_overflow(graph, conflict_signal(graph, res))
    return {
        "converged": converged_at is not None,
        "epochs": converged_at if converged_at is not None else max_epochs,
        "final_drc": int(res.drc_violations),
        "wirelength": float(res.wirelength_mm),
        "vias": int(res.via_count),
        "seconds": time.time() - t0,
    }


def _policies(checkpoint: str):
    pols = [UniformPredictor(), MockPredictor()]
    try:
        import torch
        from apricorn.stgnn_engine.predictor import STGNNPredictor

        torch.manual_seed(0)
        un = STGNNPredictor(checkpoint_path="__untrained__.pt")
        un.name = "stgnn-untrained"
        pols.append(un)
        if os.path.exists(checkpoint):
            tr = STGNNPredictor(checkpoint_path=checkpoint)
            tr.name = "stgnn-trained"
            pols.append(tr)
        else:
            print(f"[eval] no checkpoint at '{checkpoint}' - run training first; skipping stgnn-trained")
    except ImportError as e:
        print(f"[eval] torch not available ({e}); evaluating non-neural policies only")
    return pols


def evaluate(test_dir: str, generate: int, checkpoint: str, max_epochs: int, drc: str, out: str) -> Dict:
    from apricorn.data_graph.graph_builder import build_graph

    paths = list_boards(test_dir)
    if not paths and generate > 0:
        print(f"[eval] generating {generate} held-out boards in '{test_dir}' (seed range disjoint from training)")
        paths = generate_mixed_dataset(test_dir, generate, seed_offset=90000)
    if not paths:
        raise SystemExit(f"[eval] no boards in '{test_dir}'")

    drc_fn = None
    if drc == "real":
        from apricorn.data_graph.drc_engine import check_drc
        drc_fn = check_drc

    pols = _policies(checkpoint)
    per: Dict[str, List[Dict]] = {p.name: [] for p in pols}
    for i, path in enumerate(paths):
        graph = build_graph(load_spec(path))
        for p in pols:
            per[p.name].append(run_closed_loop(graph, p, drc_fn, max_epochs))
        if (i + 1) % 10 == 0:
            print(f"[eval] {i + 1}/{len(paths)} boards")

    # Wirelength / vias are only comparable on boards that EVERY policy fully solved.
    common = [i for i in range(len(paths)) if all(per[n][i]["converged"] for n in per)]
    summary = {}
    for name, rows in per.items():
        summary[name] = {
            "boards": len(rows),
            "success_rate": float(np.mean([r["converged"] for r in rows])),
            "mean_epochs": float(np.mean([r["epochs"] for r in rows])),
            "mean_final_drc": float(np.mean([r["final_drc"] for r in rows])),
            "mean_seconds": float(np.mean([r["seconds"] for r in rows])),
            "common_mean_wirelength": float(np.mean([rows[i]["wirelength"] for i in common])) if common else None,
            "common_mean_vias": float(np.mean([rows[i]["vias"] for i in common])) if common else None,
        }
    report = {"drc": drc, "max_epochs": max_epochs, "n_boards": len(paths),
              "n_common_solved": len(common), "summary": summary}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'policy':18s} {'solved':>7s} {'epochs':>7s} {'finalDRC':>9s} {'WL*':>8s} {'vias*':>6s} {'sec':>6s}")
    for name, s in summary.items():
        wl = "n/a" if s["common_mean_wirelength"] is None else f"{s['common_mean_wirelength']:.1f}"
        vi = "n/a" if s["common_mean_vias"] is None else f"{s['common_mean_vias']:.1f}"
        print(f"{name:18s} {100 * s['success_rate']:6.1f}% {s['mean_epochs']:7.2f} {s['mean_final_drc']:9.2f} {wl:>8s} {vi:>6s} {s['mean_seconds']:6.2f}")
    print(f"(* wirelength/vias averaged over the {len(common)} boards every policy solved)\n[eval] wrote {out}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare STGNN-guided routing against baselines")
    ap.add_argument("--test-dir", default="data/test")
    ap.add_argument("--generate", type=int, default=40)
    ap.add_argument("--checkpoint", default="checkpoints/stgnn_v1.pt")
    ap.add_argument("--max-epochs", type=int, default=15)
    ap.add_argument("--drc", choices=["real", "fallback"], default="real",
                    help="real = Member 1's shapely check_drc; fallback = router's built-in node/edge check")
    ap.add_argument("--out", default="checkpoints/eval_report.json")
    a = ap.parse_args()
    evaluate(a.test_dir, a.generate, a.checkpoint, a.max_epochs, a.drc, a.out)


if __name__ == "__main__":
    main()
