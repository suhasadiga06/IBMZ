"""Model 2 runtime: the relationship classifier, on IBM Snap ML where available.

On IBM z16/z17 Snap ML can run tree-ensemble inference on the on-chip
Integrated Accelerator for AI (tree_format="zdnn_tensors"), or on the CPU with
its compressed-tree engine ("compress_trees"). Which is faster depends on the
model and batch size, so the benchmark measures both and the service uses the
format given in SAFEPAY_TREE_FORMAT (default: whatever Snap ML picks, "auto").

Elsewhere (e.g. a laptop without Snap ML) a scikit-learn model is used.
"""
from __future__ import annotations

import os

import numpy as np

try:
    import snapml
except ImportError:  # pragma: no cover - laptop / x86 without Snap ML
    snapml = None


def make(kind: str = "boosting", seed: int = 0):
    if snapml is not None:
        if kind == "boosting":
            return snapml.BoostingMachineClassifier(num_round=300, max_depth=5, learning_rate=0.08,
                                                    colsample_bytree=0.8, subsample=0.9, random_state=seed,
                                                    n_jobs=3)
        return snapml.RandomForestClassifier(n_estimators=200, max_depth=10, random_state=seed, n_jobs=3)
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    if kind == "boosting":
        return HistGradientBoostingClassifier(max_iter=300, max_depth=5, learning_rate=0.08, random_state=seed)
    return RandomForestClassifier(n_estimators=200, max_depth=10, random_state=seed, n_jobs=-1)


def optimise(model, X_ref: np.ndarray, tree_format: str | None = None) -> str:
    """Pick the inference engine. Returns the format actually in use."""
    if snapml is None or not hasattr(model, "optimize_trees"):
        return "sklearn"
    fmt = tree_format or os.environ.get("SAFEPAY_TREE_FORMAT", "auto")
    model.optimize_trees(tree_format=fmt, X=np.asarray(X_ref[:256], dtype=np.float32))
    return getattr(model, "optimized_tree_format_", None) or fmt


def fit(model, X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None):
    X = np.asarray(X, dtype=np.float32)
    try:
        model.fit(X, y, sample_weight=w)
    except TypeError:
        model.fit(X, y)
    return model
