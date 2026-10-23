# Project Apricorn: STGNN-guided PCB router (merged project)

Board -> **graph** (Member 1) -> **STGNN edge costs + net priorities** (Member 2) -> **negotiated octilinear A\*** (Member 3)
-> routed board in the browser, with router conflicts fed back into the graph every epoch.

```
apricorn/
  contracts.py        locked interface (unchanged)
  mocks.py            deterministic mock graph/costs
  main.py             CLI pipeline
  data_graph/         parser, synthetic boards, graph builder, shapely/rtree DRC
  stgnn_engine/       model.py heads.py predictor.py  |  train.py evaluate.py labels.py feedback.py dataset.py
  router_ui/          astar_router.py negotiator.py  |  session.py (pipeline wiring)  server.py  static/index.html
tests/
```

## Quickstart

```bash
pip install -e ".[dev]"                       # Python >= 3.11; torch + torch_geometric must install cleanly
pytest -v                                      # everything should pass; torch tests skip if torch is absent

python -m apricorn.stgnn_engine.train          # ~300 mixed-difficulty boards, 25 epochs -> checkpoints/stgnn_v1.pt
python -m apricorn.stgnn_engine.evaluate       # held-out boards: uniform vs random vs untrained vs trained STGNN
python -m apricorn.router_ui.server            # http://localhost:8000
python -m apricorn.main                        # CLI run of the same pipeline
```

Useful switches: `--device cuda` (train), `--drc fallback` (evaluate / main) or env `APRICORN_DRC=fallback`
(server) to use the router's built-in node/edge check instead of the shapely engine.
If you change router or graph code, delete `data/rollout_cache.pkl` before retraining.

## How the model is trained (train.py)

* **Imitation:** Member 1's boards carry ground-truth routes; edges on them must get *lower* weight than other edges
  (pairwise hinge loss, so the 1.0 + softplus guarantee is untouched).
* **Congestion:** the real router is run for T epochs with neutral costs; edges it later conflicts on must get *higher*
  weight, and nets it had to rip up most often must get *higher* priority (differentiable priority head).
* The GRU is trained on those sequences with router feedback fed back exactly as at inference (`feedback.conflict_signal`
  -> `update_overflow`).
* 10% of boards are held out for validation; the best validation checkpoint is kept; `training_log.json` is what the
  dashboard's "Model Metrics" panel shows.

## Known limitations (read before the demo)

* **Torch code was written without being executed** in the environment that produced this package (no torch/network).
  Run `pytest tests/test_training_smoke.py` first; fix anything it surfaces before the long training run.
* **The real shapely DRC may be stricter than the router.** The router repairs node shorts and edge overflow. If
  `tests/test_e2e_merge.py::test_e2e_real_drc` fails, the remaining violations are clearance/diagonal-crossing cases the
  router does not detect.
* **Very dense boards are not always solved** (e.g. 2 layers, 16x16, 30 nets) even though a solution exists.
* **Random costs already help:** random mock costs break the router's tie-breaking, so they solve more boards than
  neutral costs. The trained STGNN has to beat *random-mock* in `evaluate`, not just `uniform`, to claim it helps.
