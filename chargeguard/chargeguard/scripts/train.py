"""Train and evaluate the ChargeGuard risk model on the Shenzhen charging data.

    python -m chargeguard.scripts.train [--data data/ev30.csv.gz] [--engine auto|sklearn|snapml]

Evaluation design, fixed before looking at any test result:
- Six stations that have both normal and faulty sessions are locked away as the test
  set (random choice, fixed seed). Nothing about them is used until the final scoring.
- Thresholds come only from the other stations: 5-fold cross-validation grouped by
  station gives out-of-fold scores, and each level's threshold is the score that a
  fixed share of normal sessions ever reaches (policy.BUDGETS).
- The model trains on snapshots of each session (minute 3, 5, 8, 10, ... and the last
  reading) and is scored on every minute of every test session, as the service sees it.
- A gun-temperature cut-off with the same false-action budgets is the baseline, and a
  model without vehicle-class features checks that it isn't just recognising vehicles.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from chargeguard.core import features as F
from chargeguard.core import model as M
from chargeguard.core import policy as P
from chargeguard.core.jsonsafe import clean

ROOT = Path(__file__).resolve().parents[2]
SNAP = (3, 5, 8, 10, 15, 20, 30, 45, 60, 90, 120, 180)
N_TEST_STATIONS = 6
SEED = 2026
BOOT = 1000
LEVEL_KEYS = ("watch", "slow", "stop")


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"transaction_id": str})
    df["end_time"] = pd.to_datetime(df["end_time"])
    return df.sort_values(["transaction_id", "end_time", "id"], kind="stable").reset_index(drop=True)


def cached_features(df: pd.DataFrame, data: Path) -> tuple[np.ndarray, np.ndarray]:
    cache = ROOT / "artifacts" / "features.npz"
    key = f"{data.stat().st_size}:{len(df)}:{len(F.FEATURES)}"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if str(z["key"]) == key:
            return z["X"], z["minute"]
    t = time.time()
    X, minute = F.replay(df)
    print(f"replayed {len(df):,} readings in {time.time() - t:.0f}s")
    np.savez(cache, X=X, minute=minute, key=key)
    return X, minute


class Sessions:
    """Row ranges and labels per charging session (rows are contiguous after sorting)."""

    def __init__(self, df: pd.DataFrame):
        code = pd.factorize(df["transaction_id"])[0]
        self.start = np.flatnonzero(np.r_[True, code[1:] != code[:-1]])
        self.end = np.r_[self.start[1:], len(df)]
        self.code = code
        first = df.iloc[self.start]
        self.station = first["class_judge"].to_numpy()
        self.y = first["label"].to_numpy().astype(int)
        self.types = first["types"].to_numpy()
        self.kwh = df["total_charging_kwh"].to_numpy(dtype=float)
        self.tx = first["transaction_id"].to_numpy()
        self.begin = pd.to_datetime(first["begin_time"]).to_numpy()
        self.n = len(self.start)

    def rows(self, sess: np.ndarray) -> np.ndarray:
        return np.isin(self.code, sess)


def oof(X, y_row, snap, S: Sessions, train_sess, cols, engine) -> np.ndarray:
    p = np.full(len(X), np.nan)
    for a, b in GroupKFold(5).split(train_sess, groups=S.station[train_sess]):
        fit_rows = S.rows(train_sess[a]) & snap
        val_rows = S.rows(train_sess[b])
        m = M.fit(M.make(engine), X[fit_rows][:, cols], y_row[fit_rows])
        p[val_rows] = M.proba(m, X[val_rows][:, cols])
    return p


def effective_all(p: np.ndarray, S: Sessions, sess: np.ndarray) -> dict[int, np.ndarray]:
    return {s: P.effective(p[S.start[s]:S.end[s]]) for s in sess}


def max_eff(eff: dict[int, np.ndarray]) -> np.ndarray:
    return np.array([np.nanmax(e) if np.isfinite(e).any() else 0.0 for e in eff.values()])


def _rate_ci(hit: np.ndarray, y: np.ndarray, rng) -> dict:
    """Recall, precision and false actions per 1,000 normal sessions, with session-bootstrap 95% CIs."""
    def stats(h, yy):
        tp, fp = (h & (yy == 1)).sum(), (h & (yy == 0)).sum()
        pos, neg = max((yy == 1).sum(), 1), max((yy == 0).sum(), 1)
        return tp / pos, tp / (tp + fp) if tp + fp else np.nan, 1000 * fp / neg
    point = stats(hit, y)
    idx = rng.integers(0, len(y), size=(BOOT, len(y)))
    boot = np.array([stats(hit[i], y[i]) for i in idx])
    lo, hi = np.nanpercentile(boot, 2.5, axis=0), np.nanpercentile(boot, 97.5, axis=0)
    names = ("recall", "precision", "false_per_1000_normal")
    return {n: {"value": round(float(v), 4), "ci95": [round(float(a), 4), round(float(b), 4)]}
            for n, v, a, b in zip(names, point, lo, hi)}


def evaluate(p: np.ndarray, S: Sessions, sess: np.ndarray, th: P.Thresholds, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    eff = effective_all(p, S, sess)
    y = S.y[sess]
    out: dict = {"sessions": int(len(sess)), "faulty_sessions": int(y.sum()),
                 "stations": sorted(int(s) for s in np.unique(S.station[sess]))}
    # Ranking quality of the score at fixed points in the session (every session is >= 31 minutes).
    out["roc_auc_at_minute"] = {}
    for minute in (3, 5, 10, 20):
        score = np.array([np.nan_to_num(eff[s][minute - 1], nan=0.0) for s in sess])
        out["roc_auc_at_minute"][str(minute)] = round(float(roc_auc_score(y, score)), 4)
    m = max_eff(eff)
    out["roc_auc_session"] = round(float(roc_auc_score(y, m)), 4)
    out["average_precision_session"] = round(float(average_precision_score(y, m)), 4)
    out["levels"] = {}
    for key in LEVEL_KEYS:
        thr = getattr(th, key)
        first = np.array([P.first_at_or_above(eff[s], thr) for s in sess])
        hit = first >= 0
        lv = _rate_ci(hit, y, rng)
        caught = hit & (y == 1)
        lv["median_minute_first_reached_faulty"] = float(np.median(first[caught] + 1)) if caught.any() else None
        # Energy that would still have gone into the pack after this level was reached.
        after = [S.kwh[S.end[s] - 1] - S.kwh[S.start[s] + f] for s, f in zip(sess[caught], first[caught])]
        total = [S.kwh[S.end[s] - 1] for s in sess[caught]]
        lv["faulty_kwh_after_level"] = round(float(np.sum(after)), 1)
        lv["faulty_kwh_share_after_level"] = round(float(np.sum(after) / max(np.sum(total), 1e-9)), 4)
        out["levels"][key] = lv
    return out


def breakdown(p, S: Sessions, sess, th, by: np.ndarray, names: dict | None = None) -> dict:
    res = {}
    for v in np.unique(by[sess]):
        sub = sess[by[sess] == v]
        y = S.y[sub]
        eff = effective_all(p, S, sub)
        entry = {"sessions": int(len(sub)), "faulty_sessions": int(y.sum())}
        if 0 < y.sum() < len(y):
            entry["roc_auc_session"] = round(float(roc_auc_score(y, max_eff(eff))), 4)
        rng = np.random.default_rng(SEED)
        for key in LEVEL_KEYS:
            hit = np.array([P.first_at_or_above(e, getattr(th, key)) >= 0 for e in eff.values()])
            entry[key] = _rate_ci(hit, y, rng)
        label = names.get(int(v), str(v)) if names else str(int(v))
        res[label] = entry
    return res


def station_check_replay(p, S: Sessions, sess, th: P.Thresholds) -> dict:
    """Run the service's label-free station check over the test stations, in time order.

    Same rule as Screener._station_update (policy.station_outlier): each station is compared
    with the other stations as they stand at that moment. Reports when each would be flagged.
    (Alert quality "before the flag" is not reported: the flag is triggered by the alerts
    themselves, so that subset would be biased in ChargeGuard's favour.)
    """
    eff = effective_all(p, S, sess)
    slowed = {s: bool(np.nan_to_num(eff[s], nan=-1).max() >= th.slow) for s in sess}
    recent: dict = {}
    seen: dict = {}
    flagged: dict = {}
    for s in sorted(sess, key=lambda s: S.begin[s]):
        st = int(S.station[s])
        q = (recent.get(st, []) + [slowed[s]])[-P.STATION_WINDOW:]
        recent[st] = q
        seen[st] = seen.get(st, 0) + 1
        others = [float(np.mean(o)) for k, o in recent.items() if k != st and len(o) >= P.STATION_MIN_SESSIONS]
        if st not in flagged and P.station_outlier(float(np.mean(q)), len(q), others):
            flagged[st] = (seen[st], str(S.begin[s])[:10])
    out = {str(st): {"sessions": int((S.station[sess] == st).sum()),
                     "flagged_after_sessions": flagged[st][0] if st in flagged else None,
                     "flagged_on": flagged[st][1] if st in flagged else None}
           for st in sorted(int(x) for x in np.unique(S.station[sess]))}
    return {"rule": (f"flag when >{5 * P.BUDGETS['slow']:.0%} of a station's last {P.STATION_WINDOW} sessions "
                     f"reached 'slow' (after at least {P.STATION_MIN_SESSIONS}) and that share is more than "
                     f"{P.FLEET_FACTOR:g}x the median of the other stations"), "stations": out}


def reference(X: np.ndarray, rows: np.ndarray) -> dict:
    """Normal-session percentiles per feature, used to explain decisions in plain words."""
    q = np.percentile(X[rows], [1, 5, 50, 95, 99], axis=0)
    return {f: [float(v) for v in col] for f, col in zip(F.FEATURES, q.T)}


def examples(df, p, S: Sessions, sess, th, k=3) -> list[dict]:
    eff = effective_all(p, S, sess)
    me = dict(zip(sess, max_eff(eff)))
    faulty = sorted((s for s in sess if S.y[s] == 1), key=lambda s: -me[s])
    normal = sorted((s for s in sess if S.y[s] == 0), key=lambda s: -me[s])
    picks = [("faulty_caught", faulty[0]), ("faulty_missed", faulty[-1]),
             ("normal_highest_score", normal[0]), ("normal_typical", normal[len(normal) // 2])][:k + 1]
    out = []
    for kind, s in picks:
        r = slice(S.start[s], S.end[s])
        part = df.iloc[r]
        out.append({"kind": kind, "transaction": str(S.tx[s]), "label": int(S.y[s]),
                    "battery": F.BATTERY_TYPES.get(int(S.types[s]), str(S.types[s])),
                    "score": np.round(p[r], 4).tolist(), "effective": np.round(np.nan_to_num(eff[s]), 4).tolist(),
                    "temp1": part[F.COLS["temp1"]].tolist(), "temp2": part[F.COLS["temp2"]].tolist(),
                    "current": part[F.COLS["current"]].tolist(), "voltage": part[F.COLS["voltage"]].tolist(),
                    "soc": part[F.COLS["soc"]].tolist()})
    return out


def run_model(X, y_row, snap, S, train_sess, test_sess, train_normal, cols, engine):
    t = time.time()
    p_oof = oof(X, y_row, snap, S, train_sess, cols, engine)
    eff_normal = max_eff(effective_all(p_oof, S, train_normal))
    st_normal = S.station[train_normal]
    # Step 1: robust provisional thresholds (median over stations) to spot stations whose
    # hardware makes normal charges look abnormal; they get a hardware check instead.
    th0 = P.choose(eff_normal, stations=st_normal)
    rates0 = {int(s): float((eff_normal[st_normal == s] >= th0.slow).mean()) for s in np.unique(st_normal)}
    flagged = P.station_check(rates0)
    # Step 2: final thresholds pooled over the remaining stations' normal sessions.
    keep = ~np.isin(st_normal, flagged)
    th = P.choose(eff_normal[keep])
    rates = {int(s): round(float((eff_normal[st_normal == s] >= th.slow).mean()), 4) for s in np.unique(st_normal)}
    extra = {"pooled_thresholds": P.choose(eff_normal),  # pooled over all stations, kept for the report
             "train_station_slow_rate_normal": rates, "flagged_stations": flagged}
    cv = evaluate(p_oof, S, train_sess, th)
    fit_rows = S.rows(train_sess) & snap
    final = M.fit(M.make(engine), X[fit_rows][:, cols], y_row[fit_rows])
    test_rows = S.rows(test_sess)
    p = np.full(len(X), np.nan)
    p[test_rows] = M.proba(final, X[test_rows][:, cols])
    return final, th, p, cv, time.time() - t, p_oof, extra


BANDS = {0: "under 90 kWh (typical cars)", 1: "90-150 kWh", 2: "over 150 kWh", 3: "unknown"}


def capacity_band(X: np.ndarray, S: Sessions) -> np.ndarray:
    """Apparent pack size per session (kWh per 1% of charge x 100), for stratified evaluation only."""
    cap = X[S.end - 1, F.FEATURES.index("kwh_per_soc")].astype(float) * 100
    return np.select([cap <= 0, cap < 90, cap < 150], [3, 0, 1], default=2)


def summary_line(name: str, ev: dict) -> str:
    lv = ev["levels"]
    return (f"  {name}: ROC-AUC {ev['roc_auc_session']} (minute 5: {ev['roc_auc_at_minute']['5']}), "
            + "  ".join(f"{k} recall {lv[k]['recall']['value']:.2f} prec {lv[k]['precision']['value']:.2f} "
                        f"false/1000 {lv[k]['false_per_1000_normal']['value']:.0f}" for k in LEVEL_KEYS))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "ev30.csv.gz"))
    ap.add_argument("--engine", default="auto", choices=["auto", "sklearn", "snapml"])
    ap.add_argument("--skip-comparison", action="store_true", help="skip the all-features comparison model")
    args = ap.parse_args()
    engine = M.engine_name(args.engine)
    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)

    data = Path(args.data)
    df = load(data)
    X, minute = cached_features(df, data)
    S = Sessions(df)
    y_row = df["label"].to_numpy().astype(int)
    is_last = np.zeros(len(df), bool)
    is_last[S.end - 1] = True
    snap = np.isin(minute, SNAP) | is_last
    band = capacity_band(X, S)

    st = pd.DataFrame({"station": S.station, "y": S.y}).groupby("station").y.agg(["mean", "size"])
    mixed = st[(st["mean"] > 0) & (st["mean"] < 1) & (st["size"] >= 100)].index.to_numpy()
    test_st = np.sort(np.random.default_rng(SEED).choice(mixed, N_TEST_STATIONS, replace=False))
    all_sess = np.arange(S.n)
    test_sess = all_sess[np.isin(S.station, test_st)]
    train_sess = all_sess[~np.isin(S.station, test_st)]
    train_normal = train_sess[S.y[train_sess] == 0]
    cars = test_sess[band[test_sess] == 0]
    print(f"engine={engine}  train: {len(train_sess):,} sessions ({S.y[train_sess].sum()} faulty), "
          f"test stations {test_st.tolist()}: {len(test_sess):,} sessions ({S.y[test_sess].sum()} faulty; "
          f"{len(cars):,} typical-car sessions with {S.y[cars].sum()} faulty)")

    feats = F.BEHAVIOUR_FEATURES
    cols = [F.FEATURES.index(f) for f in feats]
    final, th, p_test, cv, secs, p_oof, extra = run_model(X, y_row, snap, S, train_sess, test_sess,
                                                           train_normal, cols, engine)
    np.savez(ROOT / "artifacts" / "scores.npz", p_oof=p_oof, p_test=p_test)
    test = evaluate(p_test, S, test_sess, th)
    test_cars = evaluate(p_test, S, cars, th)
    pooled = evaluate(p_test, S, test_sess, extra["pooled_thresholds"])
    names = {int(k): v for k, v in F.BATTERY_TYPES.items()}
    print(f"behaviour-only model ({len(feats)} features): {secs:.0f}s  thresholds {th}")
    print(f"  stations flagged for a hardware check (training data): {extra['flagged_stations']}")
    print(summary_line("all test sessions", test))
    print(summary_line("typical cars", test_cars))
    check = station_check_replay(p_test, S, test_sess, th)
    print("  runtime station check on test stations:",
          {k: v["flagged_after_sessions"] for k, v in check["stations"].items()})

    # Baseline: the gun-temperature cut-off, with the same false-action budgets on training stations.
    temp = np.maximum(df[F.COLS["temp1"]].to_numpy(float), df[F.COLS["temp2"]].to_numpy(float))
    keep_base = train_normal[~np.isin(S.station[train_normal], extra["flagged_stations"])]
    th_temp = P.choose(max_eff(effective_all(temp, S, keep_base)))
    base = evaluate(temp, S, test_sess, th_temp)
    base["thresholds_celsius"] = th_temp.to_dict()
    base_cars = evaluate(temp, S, cars, th_temp)
    print(summary_line("temperature cut-off, all", base))
    print(summary_line("temperature cut-off, typical cars", base_cars))

    comparison = None
    if not args.skip_comparison:
        cols_all = list(range(len(F.FEATURES)))
        _, th_c, p_c, _, _, _, _ = run_model(X, y_row, snap, S, train_sess, test_sess, train_normal, cols_all, engine)
        comparison = {"features": "all, including vehicle-size levels", "thresholds": th_c.to_dict(),
                      "test": evaluate(p_c, S, test_sess, th_c), "test_typical_cars": evaluate(p_c, S, cars, th_c),
                      "test_by_capacity_band": breakdown(p_c, S, test_sess, th_c, band, BANDS)}
        print(summary_line("all-features model, all", comparison["test"]))
        print(summary_line("all-features model, typical cars", comparison["test_typical_cars"]))

    metrics = {
        "dataset": {"name": "Shenzhen EV charging (PolyU + Autosun), Mendeley 10.17632/c7gg94tmvz.3",
                    "paper": "Yang et al., Nature Communications 17:974 (2025), doi:10.1038/s41467-025-67703-7",
                    "readings": int(len(df)), "sessions": int(S.n), "faulty_sessions": int(S.y.sum()),
                    "stations": int(len(st)),
                    "label": "1 = charging record from a faulty battery, per the dataset authors (constant per "
                             "session). Faults, not fires: no link to fire outcomes is documented."},
        "split": {"seed": SEED, "test_stations": test_st.tolist(),
                  "train_sessions": int(len(train_sess)), "train_faulty": int(S.y[train_sess].sum()),
                  "test_sessions": int(len(test_sess)), "test_faulty": int(S.y[test_sess].sum()),
                  "test_typical_car_sessions": int(len(cars)), "test_typical_car_faulty": int(S.y[cars].sum()),
                  "stations_all_faulty_kept_in_training": st[st["mean"] == 1].index.astype(int).tolist()},
        "capacity_confounder": {
            BANDS[b]: {"sessions": int((band == b).sum()), "faulty": int(S.y[band == b].sum())} for b in BANDS},
        "engine": engine, "platform": {"machine": platform.machine(), "python": platform.python_version()},
        "features": feats, "features_left_out": sorted(F.SIZE_FEATURES),
        "budgets_share_of_normal_sessions": P.BUDGETS,
        "thresholds": th.to_dict(),
        "threshold_method": ("stations whose normal sessions reach 'slow' > 5x the budget under robust "
                             "(median-over-stations) thresholds are flagged for a hardware check; final "
                             "thresholds are pooled quantiles over the other training stations' normal sessions"),
        "train_station_slow_rate_normal": extra["train_station_slow_rate_normal"],
        "stations_flagged_for_hardware_check": extra["flagged_stations"],
        "pooled_threshold_variant": {"thresholds": extra["pooled_thresholds"].to_dict(), "test_levels": pooled["levels"],
                                     "test_by_chemistry": breakdown(p_test, S, test_sess, extra["pooled_thresholds"],
                                                                    S.types, names)},
        "cross_validation_on_training_stations": cv,
        "test": test,
        "test_typical_cars": test_cars,
        "test_by_capacity_band": breakdown(p_test, S, test_sess, th, band, BANDS),
        "test_by_chemistry": breakdown(p_test, S, test_sess, th, S.types, names),
        "test_by_station": breakdown(p_test, S, test_sess, th, S.station),
        "test_runtime_station_check": check,
        "baseline_temperature_cutoff": base,
        "baseline_temperature_cutoff_typical_cars": base_cars,
        "comparison_all_features": comparison,
    }
    (ROOT / "results" / "metrics.json").write_text(json.dumps(clean(metrics), indent=2, allow_nan=False))
    (ROOT / "results" / "examples.json").write_text(json.dumps(clean(examples(df, p_test, S, test_sess, th)),
                                                                allow_nan=False))
    keep = train_normal[~np.isin(S.station[train_normal], extra["flagged_stations"])]
    joblib.dump({"model": final, "engine": engine, "features": feats, "thresholds": th.to_dict(),
                 "test_stations": test_st.tolist(), "reference": reference(X, S.rows(train_normal) & snap),
                 "flagged_stations": extra["flagged_stations"],
                 "normal_max_scores": np.sort(max_eff(effective_all(p_oof, S, keep))).astype(np.float32),
                 "trained_on": platform.machine()}, ROOT / "artifacts" / "risk.joblib")
    print("wrote results/metrics.json, results/examples.json, artifacts/risk.joblib")


if __name__ == "__main__":
    main()
