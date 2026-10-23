# apricorn/stgnn_engine/__init__.py
# Lazy import: numpy-only helpers (feedback.py, labels.py) stay importable even
# when torch is not installed. `from apricorn.stgnn_engine import STGNNPredictor`
# still works exactly as before.
__all__ = ["STGNNPredictor"]


def __getattr__(name):
    if name == "STGNNPredictor":
        from apricorn.stgnn_engine.predictor import STGNNPredictor
        return STGNNPredictor
    raise AttributeError(name)
