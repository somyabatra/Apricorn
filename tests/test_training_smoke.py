"""Needs torch + torch_geometric. Tiny end-to-end training run: checkpoint is produced and loadable."""
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")


def test_priority_head_is_differentiable():
    from apricorn.mocks import make_mock_graph
    from apricorn.stgnn_engine.heads import ApricornHeads

    g = make_mock_graph(L=2, X=8, Y=8, nets=4)
    heads = ApricornHeads()
    h = torch.randn(g.x_nodes.shape[0], 64, requires_grad=True)
    ids, scores = heads.net_priority_head.forward_tensor(h, g.net_demands)
    assert len(ids) == len(g.net_demands) and scores.shape == (len(ids),)
    scores.sum().backward()
    assert h.grad is not None and h.grad.abs().sum() > 0
    # same numbers as the python-float forward() used at inference
    as_dict = heads.net_priority_head(h.detach(), g.net_demands, torch.device("cpu"))
    for nid, s in zip(ids, scores.tolist()):
        assert abs(as_dict[nid] - s) < 1e-5


def test_train_writes_checkpoint_log_and_predictor_loads_it(tmp_path):
    from apricorn.data_graph.graph_builder import build_graph
    from apricorn.data_graph.parser_gen import generate_synthetic_board
    from apricorn.stgnn_engine.dataset import generate_mixed_dataset
    from apricorn.stgnn_engine.predictor import STGNNPredictor
    from apricorn.stgnn_engine.train import Config, train

    data = tmp_path / "boards"
    generate_mixed_dataset(str(data), 6, mix=((2, 12, 12),))
    ckpt_dir = tmp_path / "ckpt"
    cfg = Config(epochs=2, warmup_epochs=1, T=3, accum=2, pairs=64, val_frac=0.2)
    path = train(str(data), 0, str(ckpt_dir), "cpu", cfg, cache_file=None, verbose=False)

    log = json.loads((ckpt_dir / "training_log.json").read_text())
    assert len(log["history"]) == 2 and log["n_val"] >= 1
    pred = STGNNPredictor(checkpoint_path=path, device="cpu")
    assert pred.checkpoint_loaded
    g = build_graph(generate_synthetic_board(2, 12, 12, 5, seed=5))
    c = pred.predict(g, None, 1)
    assert c.edge_weights.min() >= 1.0 and c.hidden_state.shape == (g.x_nodes.shape[0], 64)
