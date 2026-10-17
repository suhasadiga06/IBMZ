"""Risk model runtime: IBM Snap ML on IBM Z, scikit-learn elsewhere.

On z16/z17, Snap ML can run tree-ensemble inference on the on-chip
Integrated Accelerator for AI (tree_format="zdnn_tensors") or on the CPU with
its compressed-tree engine ("compress_trees"). The benchmark measures both; the
service uses CHARGEGUARD_TREE_FORMAT (default "auto": Snap ML picks).
"""
from __future__ import annotations

import os

import numpy as np

try:
    import snapml
except ImportError:  # laptop / x86 without Snap ML
    snapml = None


def engine_name(engine: str = "auto") -> str:
    if engine == "auto":
        return "snapml" if snapml is not None else "sklearn"
    return engine


def make(engine: str = "auto", seed: int = 0):
    if engine_name(engine) == "snapml":
        return snapml.BoostingMachineClassifier(num_round=400, max_depth=6, learning_rate=0.05,
                                                colsample_bytree=0.8, subsample=0.9, random_state=seed,
                                                n_jobs=os.cpu_count() or 1)
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=seed)


def balanced_weights(y: np.ndarray) -> np.ndarray:
    pos = max(int(y.sum()), 1)
    neg = max(len(y) - pos, 1)
    return np.where(y == 1, len(y) / (2 * pos), len(y) / (2 * neg)).astype(np.float32)


def fit(model, X: np.ndarray, y: np.ndarray):
    model.fit(np.asarray(X, dtype=np.float32), np.asarray(y), sample_weight=balanced_weights(np.asarray(y)))
    return model


def proba(model, X) -> np.ndarray:
    p = np.asarray(model.predict_proba(np.asarray(X, dtype=np.float32)))
    return p[:, 1] if p.ndim == 2 else p


def optimise(model, X_ref: np.ndarray, tree_format: str | None = None) -> str:
    """Pick Snap ML's inference engine. Returns the format actually in use."""
    if snapml is None or not hasattr(model, "optimize_trees"):
        return "sklearn"
    fmt = tree_format or os.environ.get("CHARGEGUARD_TREE_FORMAT", "auto")
    model.optimize_trees(tree_format=fmt, X=np.asarray(X_ref[:256], dtype=np.float32))
    return getattr(model, "optimized_tree_format_", None) or fmt
