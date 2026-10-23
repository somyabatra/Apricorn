Trained STGNN weights and run logs live here.

  python -m apricorn.stgnn_engine.train      -> stgnn_v1.pt + training_log.json
  python -m apricorn.stgnn_engine.evaluate   -> eval_report.json

The web dashboard reads training_log.json / eval_report.json for its metrics, so every
number it shows comes from a real run. Commit stgnn_v1.pt so teammates get the same model.
