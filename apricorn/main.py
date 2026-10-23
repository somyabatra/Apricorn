# FILE: apricorn/main.py — FINAL MERGE SCRIPT (Member 1 + Member 2 + Member 3)
"""
Master pipeline execution script for Project Apricorn.

Integrates:
  - Member 1: load_or_generate_board(), build_graph(), update_overflow(), check_drc()
  - Member 2: STGNNPredictor.predict()
  - Member 3: NegotiatedRouter.step_epoch()

Usage:
    # Full live pipeline (after all members merge):
    python -m apricorn.main

    # Mock mode (M2+M3 using deterministic stubs, works any time):
    python -m apricorn.main --use-mocks

    # With a custom board file:
    python -m apricorn.main --board path/to/board.json
"""

from __future__ import annotations

import argparse
from typing import Optional

from apricorn.contracts import RoutedBoardResult
from apricorn.data_graph.parser_gen import load_or_generate_board  # Member 1
from apricorn.data_graph.graph_builder import build_graph, update_overflow  # Member 1
from apricorn.data_graph.drc_engine import check_drc  # Member 1
from apricorn.stgnn_engine.feedback import conflict_signal


def run_apricorn_pipeline(
    board_path: Optional[str] = None,
    max_epochs: int = 20,
    use_mocks: bool = False,
    checkpoint_path: str = "checkpoints/stgnn_v1.pt",
    verbose: bool = True,
    drc: str = "real",
) -> RoutedBoardResult:
    """
    Execute the end-to-end negotiated PCB routing loop.

    Steps per epoch:
      1. [M1] build_graph()          → GraphDataContract
      2. [M2] STGNNPredictor.predict() → RoutingCostContract
      3. [M3] NegotiatedRouter.step_epoch() → RoutedBoardResult
      4. [M1] update_overflow()      → updated GraphDataContract for t+1

    Convergence: stops when drc_violations == 0 AND edge_overflow.sum() == 0.
    """
    # --- Step 1: Member 1 loads/generates board and builds GraphDataContract ---
    board_spec = load_or_generate_board(board_path)
    graph = build_graph(board_spec)

    if verbose:
        N = graph.x_nodes.shape[0]
        E = graph.edge_index.shape[1]
        print(f"[apricorn] Board '{graph.board_id}': N={N} nodes, E={E} edges, "
              f"{len(graph.net_demands)} nets")

    # --- Wire up M2 and M3 (live or mock) ------------------------------------
    if use_mocks:
        from apricorn.mocks import make_mock_costs

        class _MockPredictor:
            """Stub for M2 — used when --use-mocks is set."""
            def predict(self, g, h_prev, epoch):
                return make_mock_costs(g, epoch=epoch)

        class _MockRouter:
            """Stub for M3 — used when --use-mocks is set."""
            def __init__(self, drc_fn):
                self.drc_fn = drc_fn

            def step_epoch(self, g, costs, epoch):
                routed = {nid: list(terminals) for nid, terminals in g.net_demands.items()}
                drc_errs = self.drc_fn(g, routed)
                import numpy as np
                overflow = np.zeros(g.edge_index.shape[1], dtype=np.float32)
                return RoutedBoardResult(
                    epoch=epoch,
                    routed_paths=routed,
                    edge_overflow=overflow,
                    drc_violations=drc_errs,
                    wirelength_mm=100.0,
                    via_count=len(g.net_demands),
                )

        predictor = _MockPredictor()
        router = _MockRouter(drc_fn=check_drc)
        if verbose:
            print("[apricorn] Running in MOCK mode (M2+M3 stubs active)")
    else:
        from apricorn.router_ui.negotiator import NegotiatedRouter   # Member 3

        try:
            from apricorn.stgnn_engine.predictor import STGNNPredictor  # Member 2
            predictor = STGNNPredictor(checkpoint_path=checkpoint_path)
        except ImportError as e:   # torch / torch_geometric missing
            from apricorn.stgnn_engine.evaluate import MockPredictor
            predictor = MockPredictor()
            print(f"[apricorn] WARNING: {e}; using random mock costs with the REAL router")

        router = NegotiatedRouter(drc_fn=check_drc if drc == "real" else None)

    # --- Negotiation loop -----------------------------------------------------
    hidden_state = None
    last_result = None

    for epoch in range(1, max_epochs + 1):
        # M1 Graph → M2 STGNN → RoutingCostContract
        costs = predictor.predict(graph, hidden_state, epoch=epoch)
        hidden_state = costs.hidden_state

        # M1 Graph + M2 Costs → M3 Wavefront A* → RoutedBoardResult
        result = router.step_epoch(graph, costs, epoch=epoch)
        last_result = result

        if verbose:
            print(f"[apricorn] Epoch {epoch:2d}: DRC={result.drc_violations}  "
                  f"overflow={result.edge_overflow.sum():.1f}  "
                  f"wirelength={result.wirelength_mm:.1f}mm  "
                  f"vias={result.via_count}")

        # Convergence check
        if result.drc_violations == 0 and result.edge_overflow.sum() == 0:
            if verbose:
                print(f"[apricorn] Converged at epoch {epoch}! [OK]")
            break

        # M3 Overflow → M1 Graph feature update for epoch t+1
        graph = update_overflow(graph, conflict_signal(graph, result))

    return last_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Project Apricorn — STGNN PCB Router")
    parser.add_argument("--board", type=str, default=None,
                        help="Path to board JSON or .kicad_pcb file (omit for synthetic)")
    parser.add_argument("--max-epochs", type=int, default=20,
                        help="Maximum negotiation epochs (default: 20)")
    parser.add_argument("--use-mocks", action="store_true",
                        help="Use mock M2/M3 stubs (for testing without full merge)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/stgnn_v1.pt",
                        help="Path to M2 STGNN checkpoint (default: checkpoints/stgnn_v1.pt)")
    parser.add_argument("--drc", choices=["real", "fallback"], default="real",
                        help="real = shapely/rtree check_drc; fallback = router's built-in node/edge check")
    parser.add_argument("--quiet", action="store_true", help="Suppress epoch-by-epoch output")
    args = parser.parse_args()

    result = run_apricorn_pipeline(
        board_path=args.board,
        max_epochs=args.max_epochs,
        use_mocks=args.use_mocks,
        checkpoint_path=args.checkpoint,
        verbose=not args.quiet,
        drc=args.drc,
    )

    if result is not None:
        print(f"\n[apricorn] Final result -- DRC: {result.drc_violations}  "
              f"wirelength: {result.wirelength_mm:.2f}mm  vias: {result.via_count}")


if __name__ == "__main__":
    main()
