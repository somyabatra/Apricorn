# What was changed in the merge, and why

## Merge
Base = Hack-o-Octo (contracts, merged mocks, data_graph, stgnn_engine, packaging). Added from Hack-O-Octa 4.0:
router_ui/ (router, negotiator, server, dashboard) and tests/test_m3_router.py.

## Bugs fixed
| Where | Problem | Fix |
|---|---|---|
| astar_router.py | A net's trace could run through another net's pin -> pin shorts the rip-up loop could never repair | foreign terminals are hard-blocked |
| astar_router.py / negotiator.py | Node shorts (two nets on one node) produced no edge overflow, so negotiation never escalated | node-collision history raises the detour penalty each epoch |
| heads.py / train.py | Priority head returned python floats via `.item()`: **no gradient ever reached it** | added `forward_tensor`; `forward()` contract unchanged |
| train.py | Loss trained the model to copy `base_cost + congestion` (its own inputs); `ground_truth_path` unused; GRU trained on an unchanging graph | imitation + congestion losses, rollouts with real feedback, validation split |
| server.py | Used `make_mock_costs` everywhere; never called the STGNN, `build_graph`, `check_drc` or `update_overflow` | wired through `session.py`; honest `mode` flag |
| server.py | `/api/model_metrics` returned invented numbers (96.4% accuracy, 200 epochs, formula loss curve) | reads real `training_log.json`; says "not trained" otherwise |
| index.html | `handleFileUpload` was referenced but never defined (upload dead); heatmap drawn from `sin()`; hardcoded 96.4% / 200 epochs / 0.012 / "324 (100%)" / fake timer; chime always played | upload implemented; heatmap from model's edge costs; all values from the backend |
| stgnn_engine/__init__.py | importing any submodule required torch | lazy import |
| feedback | only edge overflow was fed back (blind to node shorts) | `feedback.conflict_signal` used in train, evaluate, server, main |

## Added
stgnn_engine/{train,evaluate,labels,feedback,dataset}.py, router_ui/session.py, `/api/upload_board`, `/api/eval_report`,
`/api/status`, tests (e2e gate from the plan, labels/feedback, session, training smoke test).
