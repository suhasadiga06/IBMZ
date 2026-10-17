"""Train all guard models and record honest metrics.

  injection: trained on source A (guychuk sample) plus 70% of source B's train
             split (jackhhao); threshold tuned on the other 30%; scored once on
             source B's test split, which nothing upstream has seen. A model
             trained on source A alone is reported too, to show the gap.
             Duplicates across all splits are removed first.
  trace:     trained on one seed of synthetic sessions, scored on another seed.

Writes artifacts/*.joblib and artifacts/metrics.json.
Usage: python3 -m agentguard.scripts.train
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from agentguard.core import tooltrace
from agentguard.core.injection import InjectionModel
from agentguard.core.normalize import canonicalize

ROOT = Path(__file__).resolve().parents[2]
DATA, ART = ROOT / "data", ROOT / "artifacts"


def scores(y, p, thr) -> dict:
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)), "positives": int(y.sum()),
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall": round(recall_score(y, pred, zero_division=0), 4),
        "f1": round(f1_score(y, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y, p), 4),
        "false_positive_rate": round(fp / max(fp + tn, 1), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _load(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / name).dropna()
    df = df[df.text.str.strip().astype(bool)].drop_duplicates("text").copy()
    df["key"] = df.text.map(lambda t: canonicalize(t[:2000]))
    return df.drop_duplicates("key")


def _tune(model: InjectionModel, df: pd.DataFrame) -> float:
    """Threshold with the best F1 on a calibration set from the *other* source."""
    prec, rec, thr = precision_recall_curve(df.label, model.predict_proba(df.text))
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return float(thr[np.argmax(f1[:-1])])


def train_injection() -> dict:
    a = _load("train.csv")
    b_cal = _load("source_b_calib.csv")
    test = _load("source_b_test.csv")

    # Leakage control: nothing in the test set may appear anywhere upstream.
    n_cal, n_test = len(b_cal), len(test)
    b_cal = b_cal[~b_cal.key.isin(set(a.key))]
    test = test[~test.key.isin(set(a.key) | set(b_cal.key))]
    removed = (n_cal - len(b_cal)) + (n_test - len(test))
    b_fit, b_thr = train_test_split(b_cal, test_size=0.3, stratify=b_cal.label, random_state=0)
    print(f"source A {len(a)} | B-fit {len(b_fit)} | B-threshold {len(b_thr)} | "
          f"B-test {len(test)} | {removed} duplicate rows removed")

    # Baseline: source A only. Reported so the cross-source gap is visible.
    base = InjectionModel().fit(a.text, a.label)
    base.threshold = _tune(base, b_thr)
    baseline = scores(test.label.values, base.predict_proba(test.text), base.threshold)
    print("source-A-only on B-test:", baseline)

    # Deployed: both sources. B is small, so its rows are up-weighted until the
    # two sources carry equal total weight.
    train = pd.concat([a, b_fit])
    w = np.where(np.arange(len(train)) < len(a), 1.0, len(a) / max(len(b_fit), 1))
    t0 = time.time()
    model = InjectionModel().fit(train.text, train.label, sample_weight=w)
    fit_s = time.time() - t0
    model.threshold = _tune(model, b_thr)

    t0 = time.time()
    p = model.predict_proba(test.text)
    ms_per = 1000 * (time.time() - t0) / len(test)
    deployed = scores(test.label.values, p, model.threshold)
    print("deployed on B-test:     ", deployed)

    joblib.dump(model, ART / "injection.joblib", compress=3)
    out = {
        "threshold": round(model.threshold, 4),
        "fit_seconds": round(fit_s, 1),
        "inference_ms_per_prompt": round(ms_per, 3),
        "deployed_on_untouched_test": deployed,
        "baseline_source_a_only": baseline,
        "rows": {"source_a": len(a), "b_fit": len(b_fit), "b_threshold": len(b_thr),
                 "b_test": len(test), "duplicates_removed": int(removed)},
    }
    print(json.dumps({k: out[k] for k in ("threshold", "fit_seconds", "inference_ms_per_prompt")}))
    return out


def train_trace() -> dict:
    Xtr, ytr = tooltrace.make_dataset(6000, seed=11)
    Xte, yte = tooltrace.make_dataset(2000, seed=99)
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=0, n_jobs=-1)
    clf.fit(np.array(Xtr), np.array(ytr))
    p = clf.predict_proba(np.array(Xte))[:, 1]
    joblib.dump(clf, ART / "trace.joblib", compress=3)
    out = scores(np.array(yte), p, 0.5)
    out["feature_importance"] = dict(sorted(
        zip(tooltrace.TraceFeatures.names(), map(lambda v: round(float(v), 4), clf.feature_importances_)),
        key=lambda kv: -kv[1]))
    print("trace:", {k: out[k] for k in ("precision", "recall", "f1", "false_positive_rate")})
    return out


def main() -> None:
    ART.mkdir(exist_ok=True)
    metrics = {
        "platform": {"machine": platform.machine(), "byteorder": sys.byteorder,
                     "python": platform.python_version(), "system": platform.platform()},
        "datasets": {
            "source_a": "guychuk/benign-malicious-prompt-classification (Apache-2.0)",
            "source_b": "jackhhao/jailbreak-classification (Apache-2.0)",
        },
        "injection": train_injection(),
        "trace": train_trace(),
    }
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"saved -> {ART}")


if __name__ == "__main__":
    main()
