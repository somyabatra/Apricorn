# FILE: apricorn/router_ui/session.py
"""
Server-side session logic for the web visualizer (no FastAPI import, so it is unit-testable).

Wires the whole pipeline the way main.py does:
    board spec -> build_graph -> STGNNPredictor.predict -> NegotiatedRouter.step_epoch
               -> conflict_signal -> update_overflow -> next epoch
and degrades gracefully (with an honest `mode` flag) when pieces are unavailable:
    "stgnn-trained"   torch present AND checkpoints/stgnn_v1.pt loaded
    "stgnn-untrained" torch present but no checkpoint (random weights)
    "mock"            torch missing -> random mock costs (apricorn.mocks)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from apricorn.contracts import GraphDataContract, RoutedBoardResult, RoutingCostContract
from apricorn.mocks import make_mock_costs, make_mock_graph
from apricorn.router_ui.negotiator import NegotiatedRouter

ROOT = Path(__file__).resolve().parents[2]
CKPT_PATH = Path(os.environ.get("APRICORN_CKPT", ROOT / "checkpoints" / "stgnn_v1.pt"))
LOG_PATH = ROOT / "checkpoints" / "training_log.json"
EVAL_PATH = ROOT / "checkpoints" / "eval_report.json"
MAX_NODES = 12000   # upload guard: the pure-python router is not built for huge grids

# --- optional components (each degrades independently) -----------------------
try:
    from apricorn.data_graph.parser_gen import generate_synthetic_board, parse_kicad_pcb
    from apricorn.data_graph.graph_builder import build_graph, update_overflow
    from apricorn.stgnn_engine.feedback import conflict_signal
    DATA_OK = True
except Exception as _e:  # pragma: no cover
    print(f"[session] data_graph unavailable ({_e}); falling back to mock boards")
    DATA_OK = False

# APRICORN_DRC=fallback forces the router's built-in node/edge check instead of Member 1's
# shapely engine (useful if shapely is missing, or to compare the two).
DRC_FN = None
DRC_MODE = "fallback"
if os.environ.get("APRICORN_DRC", "").lower() != "fallback":
    try:
        from apricorn.data_graph.drc_engine import check_drc
        DRC_FN = check_drc
        DRC_MODE = "shapely"
    except Exception:
        pass

PREDICTOR = None
try:
    from apricorn.stgnn_engine.predictor import STGNNPredictor
    PREDICTOR = STGNNPredictor(checkpoint_path=str(CKPT_PATH))
except Exception as _e:
    print(f"[session] STGNN unavailable ({type(_e).__name__}: {_e}); using mock costs")

PRESETS = {
    "demo_board.json":        dict(layers=2, size=32, nets=15, seed=42),
    "multi_layer_board.json": dict(layers=4, size=32, nets=25, seed=7),
    "high_density.json":      dict(layers=4, size=24, nets=25, seed=4),
}


def model_mode() -> str:
    if PREDICTOR is None:
        return "mock"
    return "stgnn-trained" if getattr(PREDICTOR, "checkpoint_loaded", False) else "stgnn-untrained"


def _read_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


class SessionState:
    def __init__(self) -> None:
        self.load_preset("demo_board.json")

    # ---- board loading -------------------------------------------------------
    def _install(self, graph: GraphDataContract) -> None:
        self.graph = graph
        self.router = NegotiatedRouter(drc_fn=DRC_FN)
        self.hidden: Optional[np.ndarray] = None
        self.current_epoch = 0
        self.costs: Optional[RoutingCostContract] = None
        self.latest_result: Optional[RoutedBoardResult] = None
        self.epoch_history: List[Dict[str, Any]] = []
        self.elapsed_s = 0.0
        self.last_error: Optional[str] = None

    def load_preset(self, name: str) -> None:
        if not DATA_OK:
            self._install(make_mock_graph())
            return
        cfg = PRESETS.get(name, PRESETS["demo_board.json"])
        spec = generate_synthetic_board(cfg["layers"], cfg["size"], cfg["size"], cfg["nets"], seed=cfg["seed"])
        self.load_spec(spec)

    def load_spec(self, spec: Dict[str, Any]) -> None:
        if not DATA_OK:
            raise ValueError("data_graph module unavailable")
        L, W, H = int(spec.get("layers", 2)), int(spec.get("width", 32)), int(spec.get("height", 32))
        if L * W * H > MAX_NODES:
            raise ValueError(f"board too large: {L}x{W}x{H}={L * W * H} grid nodes (max {MAX_NODES})")
        graph = build_graph(spec)
        if not graph.net_demands:
            raise ValueError("board has no routable nets (every net needs >= 2 valid pins)")
        self._install(graph)

    def load_uploaded(self, filename: str, content: str) -> None:
        name = filename.lower()
        if name.endswith(".kicad_pcb"):
            spec = parse_kicad_pcb(content)
        elif name.endswith(".json"):
            spec = json.loads(content)
            for k in ("layers", "width", "height", "nets"):
                if k not in spec:
                    raise ValueError(f"JSON board is missing required key '{k}'")
        else:
            raise ValueError("unsupported file type (use .json or .kicad_pcb)")
        spec.setdefault("board_id", os.path.splitext(os.path.basename(filename))[0])
        self.load_spec(spec)

    def reset(self) -> None:
        self._install(self.graph if DATA_OK else make_mock_graph())

    # ---- one negotiation epoch ------------------------------------------------
    def _predict(self) -> RoutingCostContract:
        if PREDICTOR is not None:
            try:
                return PREDICTOR.predict(self.graph, self.hidden, self.current_epoch)
            except Exception as e:                      # never kill the demo on a model error
                self.last_error = f"{type(e).__name__}: {e}"
        return make_mock_costs(self.graph, epoch=self.current_epoch)

    def step(self) -> None:
        t0 = time.time()
        self.current_epoch += 1
        self.costs = self._predict()
        self.hidden = self.costs.hidden_state
        self.latest_result = self.router.step_epoch(self.graph, self.costs, epoch=self.current_epoch)
        if DATA_OK:   # router -> graph feedback (what the temporal model is for)
            self.graph = update_overflow(self.graph, conflict_signal(self.graph, self.latest_result))
        self.elapsed_s += time.time() - t0
        r = self.latest_result
        self.epoch_history.append({
            "epoch": self.current_epoch, "wirelength_mm": r.wirelength_mm, "via_count": r.via_count,
            "drc_violations": r.drc_violations, "overflow": float(r.edge_overflow.sum()),
        })

    @property
    def converged(self) -> bool:
        r = self.latest_result
        return bool(r is not None and self.current_epoch > 0 and r.drc_violations == 0
                    and float(r.edge_overflow.sum()) == 0.0)

    def run(self, max_epochs: int) -> None:
        while self.current_epoch < max_epochs:
            self.step()
            if self.converged:
                break


state = SessionState()


# ---- serialization -----------------------------------------------------------

def cost_grid(g: GraphDataContract, costs: Optional[RoutingCostContract]) -> Optional[Dict[str, Any]]:
    """Per-cell mean predicted edge weight (averaged over outgoing edges and layers), 0..1 normalised."""
    if costs is None:
        return None
    _, X, Y = g.grid_shape
    N = g.x_nodes.shape[0]
    w = costs.edge_weights.astype(np.float64)
    sums = np.zeros(N); cnt = np.zeros(N)
    np.add.at(sums, g.edge_index[0], w)
    np.add.at(cnt, g.edge_index[0], 1.0)
    node_mean = sums / np.maximum(cnt, 1.0)
    xs = np.clip(g.x_nodes[:, 0].astype(int), 0, X - 1)
    ys = np.clip(g.x_nodes[:, 1].astype(int), 0, Y - 1)
    grid = np.zeros((X, Y)); gc = np.zeros((X, Y))
    np.add.at(grid, (xs, ys), node_mean)
    np.add.at(gc, (xs, ys), 1.0)
    grid /= np.maximum(gc, 1.0)
    lo, hi = float(grid.min()), float(grid.max())
    norm = (grid - lo) / (hi - lo) if hi > lo else np.zeros_like(grid)
    return {"values": np.round(norm, 3).tolist(), "min": lo, "max": hi}


def serialize_board_state() -> Dict[str, Any]:
    s = state
    g = s.graph
    L, X, Y = g.grid_shape
    N = int(g.x_nodes.shape[0])
    E = int(g.edge_index.shape[1])
    coords = g.x_nodes[:, 0:3].tolist()
    is_obs = (g.x_nodes[:, 4] > 0.5).tolist()
    res = s.latest_result

    nets_data = {}
    for net_id, terminals in g.net_demands.items():
        nets_data[str(net_id)] = {
            "terminals": terminals,
            "terminal_coords": [coords[t] for t in terminals],
            "priority": s.costs.net_priorities.get(net_id, 0.5) if s.costs else 0.5,
        }

    routed_paths_data = {}
    connected = 0
    if res:
        for net_id, path in res.routed_paths.items():
            routed_paths_data[str(net_id)] = {"nodes": path, "coords": [coords[n] for n in path]}
            if set(g.net_demands.get(net_id, [])) <= set(path):
                connected += 1

    obstacles_data = [{"layer": li, "polygon": poly} for li, poly in g.obstacle_polygons]

    telemetry = {
        "epoch": s.current_epoch,
        "wirelength_mm": res.wirelength_mm if res else 0.0,
        "via_count": res.via_count if res else 0,
        "drc_violations": res.drc_violations if res else 0,
        "edge_overflow": float(res.edge_overflow.sum()) if res else 0.0,
        "converged": s.converged,
        "routed_nets": connected,
        "total_nets": len(g.net_demands),
        "elapsed_s": round(s.elapsed_s, 2),
        "mode": model_mode(),
        "drc_engine": DRC_MODE,
        "model_error": s.last_error,
        "history": s.epoch_history,
    }

    heatmap_edges = []
    if s.costs is not None:
        weights = s.costs.edge_weights
        h_cong = s.router.h_congestion if s.router.h_congestion is not None else np.ones(E)
        scores = weights * h_cong
        for idx in np.argsort(scores)[-200:]:
            u, v = int(g.edge_index[0][idx]), int(g.edge_index[1][idx])
            heatmap_edges.append({"u": u, "v": v, "u_coords": coords[u], "v_coords": coords[v],
                                  "weight": float(weights[idx]), "congestion": float(h_cong[idx]),
                                  "score": float(scores[idx])})

    return {
        "board_id": g.board_id, "grid_shape": [L, X, Y], "total_nodes": N, "total_edges": E,
        "coords": coords, "is_obs": is_obs, "nets": nets_data, "routed_paths": routed_paths_data,
        "obstacles": obstacles_data, "telemetry": telemetry, "heatmap_edges": heatmap_edges,
        "cost_grid": cost_grid(g, s.costs),
    }


def model_metrics() -> Dict[str, Any]:
    """REAL training telemetry from checkpoints/training_log.json (nothing is invented)."""
    base = {"model_type": "GATv2 + GC-GRU", "node_features": 12, "edge_features": 6, "hidden_dim": 64,
            "mode": model_mode()}
    log = _read_json(LOG_PATH)
    if not log or not log.get("history"):
        return {**base, "trained": False, "training_epochs": 0, "final_loss": None, "accuracy_pct": None,
                "loss_history": [],
                "message": "No training run found. Run: python -m apricorn.stgnn_engine.train"}
    hist = log["history"]
    best = next((h for h in hist if h["epoch"] == log.get("best_epoch")), hist[-1])
    acc = best.get("val_gt_pair_acc")
    return {
        **base, "trained": bool(getattr(PREDICTOR, "checkpoint_loaded", False)),
        "training_epochs": len(hist), "best_epoch": log.get("best_epoch"),
        "final_loss": log.get("best_val_loss"),
        "accuracy_pct": None if acc is None else round(100.0 * acc, 1),
        "accuracy_label": "validation: ground-truth edges ranked cheaper than other edges",
        "conflict_pair_acc": best.get("val_conflict_pair_acc"),
        "priority_corr": best.get("val_prio_corr"),
        "n_train": log.get("n_train"), "n_val": log.get("n_val"),
        "loss_history": [{"epoch": h["epoch"], "loss": h["val_loss"]} for h in hist if h.get("val_loss") is not None],
    }


def eval_report() -> Dict[str, Any]:
    rep = _read_json(EVAL_PATH)
    return rep if rep else {"available": False, "message": "Run: python -m apricorn.stgnn_engine.evaluate"}
