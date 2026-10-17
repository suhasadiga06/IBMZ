"""Score our test world with Commonwealth Bank's production models, and compare.

Inputs (from safepay/scripts/cba/run_cba_features.py, run on any machine with
PyTorch, since it has no s390x wheels):  features_YYYYMM_lagN.csv
CBA's models (github.com/h2oai/aitd, Apache-2.0) are H2O MOJO files, scored
here with plain Java (h2o-genmodel.jar), which runs fine on s390x.

What this measures:
  1. Simulator realism: CBA's models were trained on real, expert-labelled
     abuse. If they rank our synthetic abusers above our innocent relationships,
     the abuse our simulator generates looks like real abuse to them.
  2. The same ranking metrics for SafePay on the same relationships.

Usage: python3 -m safepay.scripts.score_cba --cba-dir ~/cba --features-dir ~/cba/features
"""
from __future__ import annotations

import argparse
import json
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from safepay.core.engine import Engine
from safepay.scripts.train_rel import SIZES, replay
from safepay.sim import world as W

ROOT = Path(__file__).resolve().parents[2]
RUNS = [("202605_lag0", "CBA_AITD_Short"), ("202606_lag0", "CBA_AITD_Short"), ("202607_lag0", "CBA_AITD_Short"),
        ("202608_lag0", "CBA_AITD_Short"), ("202608_lag2", "CBA_AITD_Long")]


def model_columns(mojo: Path) -> list[str]:
    ini = zipfile.ZipFile(mojo).read("model.ini").decode()
    return ini.split("[columns]")[1].split("[domains]")[0].split()[:-1]   # last one is the label


def predict(cba: Path, mojo: Path, feats: Path, out: Path) -> pd.Series:
    """Score with CBA's MOJO, or reuse predictions made elsewhere.

    On big-endian s390x, h2o-genmodel 3.42 mis-reads the MOJO's tree bytes: a
    split on feature 33 (0x0021) is read as 8448 (0x2100) and scoring fails
    with ArrayIndexOutOfBounds. So predictions are made with the same jar and
    MOJO on x86 and copied here as pred_*.csv; this function then just reads them.
    """
    if not out.exists():
        subprocess.run(["java", "-Xmx768m", "-cp", str(cba / "h2o-genmodel.jar"), "hex.genmodel.tools.PredictCsv",
                        "--mojo", str(mojo), "--input", str(feats), "--output", str(out), "--decimal"],
                       check=True, capture_output=True)
    return pd.read_csv(out)["p1"]


def ranking(y: np.ndarray, p: np.ndarray) -> dict:
    order = np.argsort(-p)
    out = {"relationships": int(len(y)), "abusive": int(y.sum()),
           "roc_auc": round(float(roc_auc_score(y, p)), 4), "pr_auc": round(float(average_precision_score(y, p)), 4)}
    for k in (50, 100, int(y.sum())):
        out[f"precision_at_{k}"] = round(float(y[order[:k]].mean()), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cba-dir", type=Path, required=True)
    ap.add_argument("--features-dir", type=Path, required=True)
    args = ap.parse_args()

    labels = pd.read_csv(ROOT / "data" / "cba_labels.csv")
    lab = {(r.sender_id, r.receiver_id): (int(r.label), r.kind) for r in labels.itertuples()}
    result = {"source": "github.com/h2oai/aitd (Apache-2.0), models CBA_AITD_Short / CBA_AITD_Long",
              "runs": {}, "notes": []}

    best: dict[tuple, float] = defaultdict(float)
    for name, model in RUNS:
        mojo = args.cba_dir / f"{model}.zip"
        feats = pd.read_csv(args.features_dir / f"features_{name}.csv")
        need = model_columns(mojo)
        missing = [c for c in need if c not in feats.columns]
        p = predict(args.cba_dir, mojo, args.features_dir / f"features_{name}.csv", args.features_dir / f"pred_{name}.csv")
        units = list(zip(feats.sender_id, feats.receiver_id))
        y = np.array([lab.get(u, (0, "?"))[0] for u in units])
        run = ranking(y, p.values) if 0 < y.sum() < len(y) else {"relationships": len(y), "abusive": int(y.sum())}
        run.update(model=model, missing_features=missing)
        result["runs"][name] = run
        print(name, model, run, flush=True)
        if model == "CBA_AITD_Short":
            for u, v in zip(units, p.values):
                best[u] = max(best[u], float(v))

    # Relationship level over the whole period: CBA_Short's highest monthly score.
    units = list(best)
    y = np.array([lab.get(u, (0, "?"))[0] for u in units])
    p_cba = np.array([best[u] for u in units])
    result["cba_short_any_month"] = ranking(y, p_cba)

    # SafePay on exactly the same relationships.
    text = joblib.load(ROOT / "artifacts" / "text.joblib")
    rel = joblib.load(ROOT / "artifacts" / "rel.joblib")
    n_ok, n_bad, seed = SIZES["test"]
    w = W.build("test", n_ok, n_bad, seed)
    ids = {acct: f"U{i:05d}" for i, acct in enumerate(sorted(w.customers))}
    rep = replay(w, Engine.for_bundle(text, rel))
    sp = defaultdict(float)
    for r in rep["rows"]:
        u = (ids[r["unit"][0]], ids[r["unit"][1]])
        sp[u] = max(sp[u], r["rel"])
    p_sp = np.array([sp.get(u, 0.0) for u in units])
    result["safepay_same_relationships"] = ranking(y, p_sp)

    # Which abusive kinds does each model put in its top-K (K = number of abusive)?
    k = int(y.sum())
    top_cba = set(np.argsort(-p_cba)[:k])
    top_sp = set(np.argsort(-p_sp)[:k])
    per_kind = defaultdict(lambda: [0, 0, 0])
    for i, u in enumerate(units):
        lbl, kind = lab.get(u, (0, "?"))
        if lbl:
            per_kind[kind][0] += 1
            per_kind[kind][1] += i in top_cba
            per_kind[kind][2] += i in top_sp
    result["abusive_kinds_in_top_k"] = {kd: {"n": v[0], "cba": round(v[1] / v[0], 3), "safepay": round(v[2] / v[0], 3)}
                                         for kd, v in sorted(per_kind.items())}
    print("CBA any month:", result["cba_short_any_month"], flush=True)
    print("SafePay:      ", result["safepay_same_relationships"], flush=True)
    print("kinds:", result["abusive_kinds_in_top_k"], flush=True)
    result["notes"].append("CBA reported ROC AUC 0.79 (Short) / 0.80 (Long) on real one-month out-of-sample data, "
                           "and 35 of its top 50 real relationships confirmed abusive.")
    (ROOT / "artifacts" / "cba_validation.json").write_text(json.dumps(result, indent=2))
    print("saved artifacts/cba_validation.json", flush=True)


if __name__ == "__main__":
    main()
