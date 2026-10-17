"""Train Model 2 (relationship risk) and compare it with a CBA-style monthly batch.

Three independent synthetic worlds:
  train  Civil Comments train text, phrase half A, seed 1    -> fit models
  val    Civil Comments validation text, phrase half A, seed 2 -> pick model, tune thresholds
  test   Civil Comments test text, phrase half B, seed 3     -> report once

Units are directed relationships (sender -> recipient). A case is "opened" the
first time the relationship score crosses the tuned threshold.

Baseline: the approach in Commonwealth Bank's published model (arXiv 2303.08016):
monthly aggregates per relationship (counts, amounts, timing, message-score
statistics, simple text statistics, reciprocity, previous-month lag), random
forest, scored at each month end. It uses *our* message model, so the
comparison isolates real-time + relationship design, not the text model.

Usage: python3 -m safepay.scripts.train_rel
"""
from __future__ import annotations

import calendar
import json
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score

from safepay.core import relmodel
from safepay.core.engine import Engine
from safepay.core.policy import Thresholds, content_flag, opens_case
from safepay.core.state import FEATURES
from safepay.sim import world as W

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"
# Test seeds 3 and 4 were used while diagnosing generalisation problems (see
# README), so final numbers come from seed 5, which no decision is based on.
SIZES = {"train": (4000, 450, 1), "val": (3000, 300, 2), "test": (4000, 400, 5)}


# -- replay a world through the engine -----------------------------------------
def replay(world: W.World, engine: Engine) -> dict:
    rels = {r.rel_id: r for r in world.relationships}
    abusive_units = {(r.sender, r.recipient): r for r in world.relationships if r.label}
    kinds = {}
    for r in world.relationships:
        kinds[(r.sender, r.recipient)] = r.kind
        kinds.setdefault((r.recipient, r.sender), r.kind + "_reverse" if r.label else r.kind)
    hides = list(world.hides)
    rows = []          # per payment: unit, ts, features, label, obs fields
    hi = 0
    for tx in world.txs:
        while hi < len(hides) and hides[hi][0] <= tx.ts:
            when, recipient, sender = hides[hi]
            engine.hide(recipient, sender, when.timestamp())
            hi += 1
        obs = engine.observe(tx.ts, tx.sender, tx.recipient, tx.amount, tx.reference)
        unit = (tx.sender, tx.recipient)
        rows.append({"unit": unit, "ts": tx.ts, "x": obs.features, "y": int(unit in abusive_units), "n": obs.state.n_tx,
                     "rel": obs.rel_score, "harm": obs.harm, "threat": obs.threat,
                     "split_harm": obs.split_harm, "split_threat": obs.split_threat,
                     "ref": tx.reference, "amount": tx.amount, "hidden": obs.state.hidden_at > 0
                     and tx.ts.timestamp() >= obs.state.hidden_at})
    return {"rows": rows, "abusive": abusive_units, "kinds": kinds, "rels": rels}


def unit_weights(rows) -> np.ndarray:
    count = defaultdict(int)
    for r in rows:
        count[r["unit"]] += 1
    return np.array([1.0 / count[r["unit"]] for r in rows], dtype=np.float32)


# -- CBA-style monthly baseline ----------------------------------------------
def month_end(ts: datetime) -> datetime:
    last = calendar.monthrange(ts.year, ts.month)[1]
    return datetime(ts.year, ts.month, last, 23, 59, 59, tzinfo=timezone.utc)


def monthly_rows(world: W.World, text_engine: Engine) -> list[dict]:
    """One row per (relationship, calendar month), CBA-style aggregates + lag."""
    abusive = {(r.sender, r.recipient) for r in world.relationships if r.label}
    by = defaultdict(list)
    for tx in world.txs:
        by[((tx.sender, tx.recipient), month_end(tx.ts))].append(tx)
    replies = defaultdict(int)
    for (unit, m), txs in by.items():
        replies[((unit[1], unit[0]), m)] += len(txs)

    def agg(txs):
        if not txs:
            return [0.0] * 16
        amt = np.array([t.amount for t in txs])
        sc = np.array([text_engine.score_text(t.reference) for t in txs])
        harm, threat = sc[:, 0], sc[:, 1]
        refs = [t.reference for t in txs]
        return [len(txs), len({t.ts.date() for t in txs}), amt.mean(), amt.min(), amt.max(), np.median(amt),
                np.mean([t.ts.hour for t in txs]), harm.mean(), harm.max(), harm.min(), np.median(harm),
                threat.mean(), threat.max(), np.mean([len(r) for r in refs]),
                np.mean([sum(c.isupper() for c in r) / max(len(r), 1) for r in refs]),
                np.mean([sum(not c.isalnum() and not c.isspace() for c in r) / max(len(r), 1) for r in refs])]

    out = []
    for (unit, m), txs in by.items():
        prev_m = month_end(m.replace(day=1) - timedelta(days=1))
        lag = agg(by.get((unit, prev_m), []))
        x = agg(txs) + [replies.get((unit, m), 0)] + lag
        out.append({"unit": unit, "month": m, "x": x, "y": int(unit in abusive)})
    return out


# -- evaluation ------------------------------------------------------------------
def tune(unit_scores: dict, labels: dict) -> float:
    units = list(unit_scores)
    y = np.array([labels.get(u, 0) for u in units])
    p = np.array([unit_scores[u] for u in units])
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return float(thr[np.argmax(f1[:-1])])


def evaluate(first_flag: dict, units: set, abusive: dict, kinds: dict, first_abuse: dict) -> dict:
    y = {u: int(u in abusive) for u in units}
    tp = sum(1 for u in units if y[u] and u in first_flag)
    fp = sum(1 for u in units if not y[u] and u in first_flag)
    fn = sum(1 for u in units if y[u] and u not in first_flag)
    n_innocent = sum(1 for u in units if not y[u])
    delays = [(first_flag[u] - first_abuse[u]).total_seconds() / 3600
              for u in units if y[u] and u in first_flag and first_abuse.get(u)]
    per_kind = defaultdict(lambda: [0, 0])
    for u in units:
        k = kinds.get(u, "other")
        per_kind[k][0] += 1
        per_kind[k][1] += u in first_flag
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {
        "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
        "abusive_relationships": tp + fn, "caught": tp, "innocent_relationships": n_innocent, "false_alarms": fp,
        "false_alarms_per_10k_innocent": round(10_000 * fp / max(n_innocent, 1), 1),
        "delay_hours_median": round(float(np.median(delays)), 1) if delays else None,
        "delay_hours_p90": round(float(np.percentile(delays, 90)), 1) if delays else None,
        "caught_within_24h_share": round(sum(d <= 24 for d in delays) / max(tp + fn, 1), 4),
        "flag_rate_by_kind": {k: round(v[1] / v[0], 4) for k, v in sorted(per_kind.items())},
    }


def first_flags(rows, t: Thresholds) -> dict:
    """When each relationship's case would open, under the live service's rule."""
    out = {}
    for r in rows:
        if r["unit"] in out:
            continue
        content = content_flag(t, r["harm"], r["threat"], r["split_harm"], r["split_threat"])
        if opens_case(t, r["rel"], r["n"], content):
            out[r["unit"]] = r["ts"]
    return out


def message_outcomes(rows, flags: dict, t_harm: float, t_threat: float) -> dict:
    """Which messages would have been hidden from the recipient, as they arrived."""
    hidden = {1: [0, 0], 0: [0, 0]}
    for r in rows:
        if not r["ref"]:
            continue
        flagged = r["unit"] in flags and r["ts"] >= flags[r["unit"]]
        hide = (r["harm"] >= t_harm or r["threat"] >= t_threat or r["split_harm"] >= t_harm
                or r["split_threat"] >= t_threat or flagged or r["hidden"])
        hidden[r["y"]][0] += 1
        hidden[r["y"]][1] += hide
    return {"abusive_messages": hidden[1][0], "abusive_hidden_share": round(hidden[1][1] / max(hidden[1][0], 1), 4),
            "innocent_messages": hidden[0][0], "innocent_hidden_share": round(hidden[0][1] / max(hidden[0][0], 1), 4)}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0, help="shrink worlds for a quick smoke test")
    ap.add_argument("--out", default="", help="suffix for output files (smoke tests)")
    args = ap.parse_args()
    for k, (a, b, s) in list(SIZES.items()):
        SIZES[k] = (max(int(a * args.scale), 20), max(int(b * args.scale), 10), s)
    text = joblib.load(ART / "text.joblib")
    t_harm, t_threat = text.thresholds["harm"], text.thresholds["threat"]
    worlds = {}
    for split, (n_ok, n_bad, seed) in SIZES.items():
        t0 = time.time()
        pool_split = "train" if split == "train" else ("val" if split == "val" else "test")
        worlds[split] = W.build(pool_split, n_ok, n_bad, seed)
        print(f"{split} world: {len(worlds[split].relationships)} relationships, "
              f"{len(worlds[split].txs)} payments ({time.time() - t0:.1f}s)", flush=True)

    # Feature replay without a relationship model.
    rep = {}
    for split, w in worlds.items():
        t0 = time.time()
        rep[split] = replay(w, Engine(text))
        print(f"replayed {split}: {len(rep[split]['rows'])} payments in {time.time() - t0:.1f}s", flush=True)

    tr = rep["train"]["rows"]
    Xtr = np.array([r["x"] for r in tr], dtype=np.float32)
    ytr = np.array([r["y"] for r in tr])
    wtr = unit_weights(tr)
    wtr = np.where(ytr == 1, wtr * (1 - ytr.mean()) / max(ytr.mean(), 1e-6), wtr).astype(np.float32)

    # Model selection on the validation world.
    best, results = None, {}
    for kind in ("boosting", "forest"):
        t0 = time.time()
        m = relmodel.fit(relmodel.make(kind), Xtr, ytr, wtr)
        fit_s = time.time() - t0
        Xva = np.array([r["x"] for r in rep["val"]["rows"]], dtype=np.float32)
        pva = m.predict_proba(Xva)[:, 1]
        unit_max = defaultdict(float)
        for r, p in zip(rep["val"]["rows"], pva):
            unit_max[r["unit"]] = max(unit_max[r["unit"]], float(p))
        labels = {u: int(u in rep["val"]["abusive"]) for u in unit_max}
        thr = tune(unit_max, labels)
        y = np.array([labels[u] for u in unit_max]); p = np.array(list(unit_max.values()))
        results[kind] = {"val_auc": round(roc_auc_score(y, p), 4),
                         "val_f1": round(f1_score(y, p >= thr), 4), "threshold": thr, "fit_seconds": round(fit_s, 1)}
        print(kind, results[kind], flush=True)
        if best is None or results[kind]["val_f1"] > results[best[0]]["val_f1"]:
            best = (kind, m, thr)
    kind, model, t_case = best

    # Final, once, on the test world: real-time engine with the chosen model.
    test_w = worlds["test"]
    # Save before optimising: Snap ML discards the original trees once they are
    # converted for an inference engine, so only the unconverted model can be
    # re-optimised later (the service and benchmark do that on load).
    joblib.dump({"model": model, "kind": kind, "threshold": t_case, "features": FEATURES},
                ART / f"rel{args.out}.joblib")
    t0 = time.time()
    fmt = relmodel.optimise(model, Xtr)
    live = replay(test_w, Engine(text, model))
    replay_s = time.time() - t0
    units = {r["unit"] for r in live["rows"]}
    first_abuse = {(r.sender, r.recipient): r.first_abuse for r in test_w.relationships if r.label}
    ours_flags = first_flags(live["rows"], Thresholds(t_harm, t_threat, t_case))
    ours = evaluate(ours_flags, units, live["abusive"], live["kinds"], first_abuse)
    ours["messages"] = message_outcomes(live["rows"], ours_flags, t_harm, t_threat)
    print("SafePay (real-time):", {k: v for k, v in ours.items() if k != "flag_rate_by_kind"}, flush=True)

    # Baseline: monthly batch.
    te = Engine(text)
    mtr = monthly_rows(worlds["train"], te)
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=0,
                                n_jobs=-1).fit(np.array([r["x"] for r in mtr]), np.array([r["y"] for r in mtr]))
    mva = monthly_rows(worlds["val"], te)
    pva = rf.predict_proba(np.array([r["x"] for r in mva]))[:, 1]
    umax = defaultdict(float)
    for r, p in zip(mva, pva):
        umax[r["unit"]] = max(umax[r["unit"]], float(p))
    b_thr = tune(umax, {u: int(u in rep["val"]["abusive"]) for u in umax})
    mte = sorted(monthly_rows(test_w, te), key=lambda r: r["month"])
    pte = rf.predict_proba(np.array([r["x"] for r in mte]))[:, 1]
    b_flags = {}
    for r, p in zip(mte, pte):
        if p >= b_thr and r["unit"] not in b_flags:
            b_flags[r["unit"]] = r["month"]
    base = evaluate(b_flags, units, live["abusive"], live["kinds"], first_abuse)
    print("CBA-style monthly:  ", {k: v for k, v in base.items() if k != "flag_rate_by_kind"}, flush=True)

    importances = getattr(model, "feature_importances_", None)
    metrics = {
        "worlds": {s: {"relationships": len(w.relationships), "payments": len(w.txs),
                       "abusive": sum(r.label for r in w.relationships)} for s, w in worlds.items()},
        "model_selection": results, "chosen": kind, "case_threshold": round(t_case, 4),
        "tree_format": fmt, "test_replay_seconds": round(replay_s, 1),
        "safepay_realtime": ours, "cba_style_monthly": base,
        "feature_importance": dict(sorted(zip(FEATURES, map(float, importances)), key=lambda kv: -kv[1]))
        if importances is not None else None,
    }
    (ART / f"metrics_rel{args.out}.json").write_text(json.dumps(metrics, indent=2, default=str))
    print("saved", flush=True)


if __name__ == "__main__":
    main()
