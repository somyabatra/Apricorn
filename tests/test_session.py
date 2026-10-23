"""Server session logic (no FastAPI needed): real pipeline wiring + honest metrics."""
import json

from apricorn.router_ui import session as S


def test_default_board_and_epoch_zero():
    S.state.load_preset("demo_board.json")
    b = S.serialize_board_state()
    assert len(b["nets"]) == 15 and b["telemetry"]["epoch"] == 0
    assert b["telemetry"]["mode"] in ("mock", "stgnn-untrained", "stgnn-trained")
    assert b["cost_grid"] is None


def test_step_and_run_produce_real_telemetry():
    S.state.load_preset("high_density.json")
    S.state.step()
    b = S.serialize_board_state()
    t = b["telemetry"]
    assert t["epoch"] == 1 and t["wirelength_mm"] > 0 and len(b["routed_paths"]) == t["total_nets"]
    cg = b["cost_grid"]
    X, Y = b["grid_shape"][1:]
    assert len(cg["values"]) == X and len(cg["values"][0]) == Y
    S.state.run(15)
    t = S.serialize_board_state()["telemetry"]
    assert t["converged"] and t["routed_nets"] == t["total_nets"] and t["drc_violations"] == 0
    json.dumps(S.serialize_board_state())          # must be JSON serialisable


def test_upload_validation():
    from apricorn.data_graph.parser_gen import generate_synthetic_board

    spec = generate_synthetic_board(2, 16, 16, 6, seed=3)
    S.state.load_uploaded("mine.json", json.dumps(spec))
    assert len(S.state.graph.net_demands) == 6
    for name, content in [("x.txt", "a"), ("y.json", "{}"),
                          ("big.json", json.dumps({**spec, "width": 300, "height": 300}))]:
        try:
            S.state.load_uploaded(name, content)
        except ValueError:
            continue
        raise AssertionError(f"{name} should have been rejected")


def test_metrics_never_invented(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "LOG_PATH", tmp_path / "missing.json")
    m = S.model_metrics()
    assert m["trained"] is False and m["accuracy_pct"] is None and m["loss_history"] == []
    log = {"best_epoch": 2, "best_val_loss": 0.25, "n_train": 9, "n_val": 1,
           "history": [{"epoch": 1, "val_loss": 0.5, "val_gt_pair_acc": 0.6},
                       {"epoch": 2, "val_loss": 0.25, "val_gt_pair_acc": 0.75}]}
    p = tmp_path / "log.json"
    p.write_text(json.dumps(log))
    monkeypatch.setattr(S, "LOG_PATH", p)
    m = S.model_metrics()
    assert m["training_epochs"] == 2 and m["final_loss"] == 0.25 and m["accuracy_pct"] == 75.0
