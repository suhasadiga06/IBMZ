"""Train and evaluate the ChargeGuard vehicle health model on the Zhang et al. battery data.

    python -m chargeguard.scripts.train_vehicle [--data ~/evdata/vehicle] [--raw ~/evdata] [--repeats 20]

The session screener (train.py) scores one charge minute by minute. This model looks across
a vehicle's charging history instead: features of each 128-step charging snippet (cell-voltage
spread, its growth during the charge, temperatures, how fast the highest cell rises) are
summarised per car, ordered by mileage (levels, upper tails, trend with mileage, share of
unusually wide-spread snippets), and a small regularised model ranks the cars.

Data: Zhang et al., "Realistic fault detection of Li-ion battery via dynamical deep learning",
Nature Communications 14:5940 (2023), doi:10.1038/s41467-023-41226-5; figshare
10.6084/m9.figshare.23659323, brands 2 and 3, converted to .npz by convert_vehicle.py.

Evaluation design (features, model settings, folds and flag budget were fixed before any result
was seen; the last two checks below were added after a first run, to see what drives the signal):
- Everything is at the car level with cars held out: repeated stratified 5-fold
  cross-validation over cars (folds stratified by brand x label), within each brand and on
  both brands pooled.
- Per-brand robust scaling and the "high spread" level are fitted on the training cars of
  each fold only. The flag threshold is the score that 10% of the normal training cars reach,
  taken from out-of-fold scores of an inner 4-fold split of the training cars.
- ROC-AUC is averaged over repeats; 95% CIs come from a bootstrap over cars (stratified by
  label). Recall, precision and false-flag rate use each car's share of repeats in which it
  was flagged.
- Cross-brand: train on all cars of one brand, test on the other (strict: the training
  brand's scaling and level; variant: the target brand scaled by its own cars, no labels).
- Early warning: each held-out car described only by snippets from the first half of its
  mileage history, scored by models trained on full histories and by models trained on
  first halves.
- Baselines: the median cell-voltage spread alone, and a usage-only model (amount of data,
  mileage, current, SOC) that shows how far the labels can be told apart without looking at
  battery behaviour at all. A check with every car cut to at most 100 snippets (evenly spread
  over its history) shows whether the model leans on how much data a car has.
- Added after the first run: a model on the cell-voltage spread features only (the drift
  hypothesis by itself), and a mileage-adjusted check (every car feature minus its per-brand
  linear trend on mileage, fitted on training cars) because faulty and normal cars differ in age.
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import time
import warnings
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import mannwhitneyu, rankdata, spearmanr
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[2]
COLUMNS = ["volt", "current", "soc", "max_single_volt", "min_single_volt", "max_temp", "min_temp", "timestamp"]
VOLT, CUR, SOC, VMAX, VMIN, TMAX, TMIN, TIME = range(8)
BRANDS = (3, 2)
SEED = 2026
FOLDS, INNER_FOLDS, REPEATS = 5, 4, 20
BOOT = 2000
BUDGET = 0.10          # share of normal training cars allowed to reach the flag
CLIP = 5.0             # scaled features are clipped to +-5 robust units
EARLY_SHARE = 0.5      # early warning: first half of each car's mileage history
MIN_SNIPPETS = 10      # cars with less history are left out (and listed)
CAP = 100              # history-size check: at most this many snippets per car
CHUNK = 4096
EXAMPLE_POINTS = 300

SNIPPET_FEATURES = ["spread_mv", "spread_max_mv", "spread_growth_mv", "spread_mid_soc_mv", "temp_spread_c",
                    "temp_max_c", "temp_rise_c", "vmax_rise_mv_per_min", "current_abs_a", "soc_start", "soc_range"]
SN = {f: i for i, f in enumerate(SNIPPET_FEATURES)}
BEHAVIOUR = ["log_spread_median", "log_spread_p95", "log_spread_mid_soc", "spread_iqr_mv", "spread_trend_rho",
             "spread_late_minus_early_mv", "spread_growth_median_mv", "temp_spread_median_c", "temp_max_p95_c",
             "vmax_rise_p95_mv_per_min"]
USAGE = ["log_snippets", "mileage_median", "mileage_span", "current_abs_median_a", "soc_start_median",
         "soc_range_median"]
CAR_FEATURES = BEHAVIOUR + USAGE + ["high_spread_share"]
MODEL_FEATURES = BEHAVIOUR + ["high_spread_share"]
SPREAD_FEATURES = [f for f in MODEL_FEATURES if "spread" in f and not f.startswith("temp")]
MODELS = {"logistic": MODEL_FEATURES, "boosting": MODEL_FEATURES, "spread_only_logistic": SPREAD_FEATURES,
          "spread_median_only": ["log_spread_median"], "usage_only_logistic": USAGE}
MIL = CAR_FEATURES.index("mileage_median")
DESCRIPTIONS = {
    "log_spread_median": "log of the median (over snippets) of the mean max-min cell-voltage spread, mV",
    "log_spread_p95": "log of the 95th percentile of the per-snippet maximum spread, mV",
    "log_spread_mid_soc": "log of the median spread while SOC is 40-80% (less dependent on where charging starts)",
    "spread_iqr_mv": "interquartile range of the per-snippet mean spread, mV",
    "spread_trend_rho": "Spearman correlation of snippet spread with mileage (drift over the history)",
    "spread_late_minus_early_mv": "median spread of the last 30% of snippets minus the first 30%, mV",
    "spread_growth_median_mv": "median growth of the spread within a snippet (last quarter minus first), mV",
    "temp_spread_median_c": "median of the mean max-min temperature difference, C",
    "temp_max_p95_c": "95th percentile of the snippet maximum temperature, C",
    "vmax_rise_p95_mv_per_min": "95th percentile of the rise rate of the highest cell voltage, mV/min",
    "high_spread_share": "share of snippets whose max spread is above the brand's high-spread level "
                         "(median over training cars of each car's 95th percentile)",
    "log_snippets": "log number of snippets (amount of history)",
    "mileage_median": "median odometer value (brand-specific units)",
    "mileage_span": "odometer range covered by the history (brand-specific units)",
    "current_abs_median_a": "median charging current, A",
    "soc_start_median": "median SOC at the start of a snippet, %",
    "soc_range_median": "median SOC gained within a snippet, %",
}


def r4(x) -> float | None:
    x = float(x)
    return round(x, 4) if np.isfinite(x) else None


def clean(o):
    """Plain JSON values: numpy scalars to Python, NaN/inf to null."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, np.ndarray)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer, np.bool_)):
        return o.item()
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


# ---------------------------------------------------------------- snippets

def iter_snippets(path: Path, chunk: int = CHUNK):
    """Yield the raw (k, 128, 8) arrays of an .npz in chunks, so brand 2 never sits in memory whole."""
    with zipfile.ZipFile(path) as z, z.open("X.npy") as f:
        version = np.lib.format.read_magic(f)
        read_header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = read_header(f)
        assert not fortran and shape[1:] == (128, len(COLUMNS)), shape
        step = int(np.prod(shape[1:])) * dtype.itemsize
        for start in range(0, shape[0], chunk):
            k = min(chunk, shape[0] - start)
            yield np.frombuffer(f.read(k * step), dtype).reshape((k,) + shape[1:])


def snippet_features(X: np.ndarray) -> np.ndarray:
    """Features of each 128-step (10 s per step) charging snippet; X has shape (n, 128, 8)."""
    X = np.asarray(X, dtype=np.float64)
    spread = (X[:, :, VMAX] - X[:, :, VMIN]) * 1000
    soc, tmax = X[:, :, SOC], X[:, :, TMAX]
    q = X.shape[1] // 4
    mid = (soc >= 40) & (soc <= 80)
    with np.errstate(invalid="ignore", divide="ignore"):
        spread_mid = (spread * mid).sum(1) / mid.sum(1)  # NaN when the snippet is never at 40-80% SOC
    t = X[:, :, TIME] - X[:, :, TIME].mean(1, keepdims=True)
    v = X[:, :, VMAX] - X[:, :, VMAX].mean(1, keepdims=True)
    rise = (t * v).sum(1) / np.maximum((t * t).sum(1), 1e-9) * 60_000  # least-squares slope, mV per minute
    return np.column_stack([
        spread.mean(1), spread.max(1), spread[:, -q:].mean(1) - spread[:, :q].mean(1), spread_mid,
        (tmax - X[:, :, TMIN]).mean(1), tmax.max(1), tmax[:, -q:].mean(1) - tmax[:, :q].mean(1),
        rise, np.abs(X[:, :, CUR]).mean(1), soc[:, 0], soc.max(1) - soc.min(1)])


class Checks:
    """Value checks behind the column mapping, accumulated while streaming the raw arrays."""

    def __init__(self):
        self.n = self.steps = 0
        self.cell_order = self.temp_order = self.current_nonpos = self.soc_nondecreasing = 0
        self.lo, self.hi = np.full(len(COLUMNS), np.inf), np.full(len(COLUMNS), -np.inf)
        self.dt: dict[float, int] = {}
        self.ratio: list[np.ndarray] = []

    def add(self, X: np.ndarray) -> None:
        self.n += len(X)
        self.steps += X.shape[0] * X.shape[1]
        self.cell_order += int((X[:, :, VMAX] >= X[:, :, VMIN]).sum())
        self.temp_order += int((X[:, :, TMAX] >= X[:, :, TMIN]).sum())
        self.current_nonpos += int((X[:, :, CUR] <= 0).sum())
        self.soc_nondecreasing += int((np.diff(X[:, :, SOC], axis=1) >= 0).sum())
        self.lo = np.minimum(self.lo, X.min((0, 1)))
        self.hi = np.maximum(self.hi, X.max((0, 1)))
        u, c = np.unique(np.diff(X[:, :, TIME], axis=1), return_counts=True)
        for a, b in zip(u.tolist(), c.tolist()):
            self.dt[a] = self.dt.get(a, 0) + b
        cell = (X[:, :, VMAX].astype(np.float64) + X[:, :, VMIN]) / 2
        self.ratio.append(np.median(X[:, :, VOLT] / cell, axis=1))

    def result(self) -> dict:
        diffs = self.n * 127
        top = sorted(self.dt.items(), key=lambda kv: -kv[1])[:3]
        return {"snippets": self.n,
                "share_max_cell_ge_min_cell": r4(self.cell_order / self.steps),
                "share_max_temp_ge_min_temp": r4(self.temp_order / self.steps),
                "share_current_le_0_while_charging": r4(self.current_nonpos / self.steps),
                "share_soc_non_decreasing": r4(self.soc_nondecreasing / diffs),
                "timestamp_step_seconds_top": {f"{a:g}": r4(b / diffs) for a, b in top},
                "ranges": {c: [r4(a), r4(b)] for c, a, b in zip(COLUMNS, self.lo, self.hi)},
                "volt_over_mean_cell_voltage_pct_1_50_99": [r4(v) for v in
                                                           np.percentile(np.concatenate(self.ratio), [1, 50, 99])]}


def load_brand(data: Path, brand: int) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Snippet metadata, snippet features and column checks for one brand (features cached)."""
    path = data / f"brand{brand}.npz"
    z = np.load(path, allow_pickle=False)
    assert [str(c) for c in z["columns"]] == COLUMNS, z["columns"]
    meta = pd.DataFrame({"car": z["car"].astype(int), "snippet_label": z["label"].astype(str),
                         "segment": pd.to_numeric(z["charge_segment"], errors="coerce"),
                         "mileage": z["mileage"].astype(float),
                         "order": pd.to_numeric(pd.Series(z["snippet"]).str.split("/").str[-1], errors="coerce")})
    cache = data / f"brand{brand}_snippet_features.npz"
    key = f"{path.stat().st_size}:{path.stat().st_mtime_ns}:{','.join(SNIPPET_FEATURES)}"
    if cache.exists():
        c = np.load(cache, allow_pickle=False)
        if str(c["key"]) == key:
            return meta, c["F"], json.loads(str(c["checks"]))
    t, checks, parts = time.time(), Checks(), []
    for X in iter_snippets(path):
        checks.add(X)
        parts.append(snippet_features(X))
    F, chk = np.concatenate(parts), checks.result()
    print(f"brand {brand}: features of {len(F):,} snippets in {time.time() - t:.0f}s")
    np.savez(cache, F=F, checks=json.dumps(chk), key=key)
    return meta, F, chk


def load_labels(raw: Path, brand: int) -> pd.DataFrame:
    files = sorted((raw / f"battery_brand{brand}" / "label").glob("*.csv"))
    lab = pd.concat([pd.read_csv(f) for f in files])[["car", "label"]].drop_duplicates()
    assert lab.car.is_unique, f"brand {brand}: conflicting labels"
    return lab.assign(label_file=",".join(f.name for f in files))


class Fleet:
    """All snippets of the chosen brands, ordered by brand, car and mileage, with per-car labels."""

    def __init__(self, parts: list[tuple[int, pd.DataFrame, np.ndarray, pd.DataFrame]]):
        meta = pd.concat([m.assign(brand=b) for b, m, _, _ in parts], ignore_index=True)
        F = np.concatenate([f for _, _, f, _ in parts])
        labels = pd.concat([lab.assign(brand=b) for b, _, _, lab in parts], ignore_index=True)
        o = np.lexsort((meta.order, meta.segment.fillna(-1), meta.mileage, meta.car, meta.brand))
        meta, F = meta.iloc[o].reset_index(drop=True), F[o]
        size = meta.groupby(["brand", "car"], sort=False).size()
        self.too_short = {f"{b}:{c}": int(n) for (b, c), n in size.items() if n < MIN_SNIPPETS}
        keep = meta.set_index(["brand", "car"]).index.isin(size[size >= MIN_SNIPPETS].index)
        meta, F = meta[keep].reset_index(drop=True), F[keep]
        key = meta.brand.to_numpy() * 100_000 + meta.car.to_numpy()
        self.start = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
        self.end = np.r_[self.start[1:], len(meta)]
        first = meta.iloc[self.start]
        self.brand, self.car = first.brand.to_numpy(), first.car.to_numpy()
        lab = labels.set_index(["brand", "car"]).label
        idx = pd.MultiIndex.from_arrays([self.brand, self.car])
        missing = ~idx.isin(lab.index)
        assert not missing.any(), f"cars without a label: {idx[missing].tolist()}"
        self.y = lab.loc[idx].to_numpy().astype(int)
        # The snippet files carry their own label ('10' faulty, '00' normal); it must agree with the car label.
        snip_y = (meta.snippet_label.str[0] == "1").to_numpy().astype(int)
        self.snippet_label_agrees = float((snip_y == np.repeat(self.y, self.end - self.start)).mean())
        self.labels, self.meta, self.F = labels, meta, F
        self.mileage = meta.mileage.to_numpy()
        self.n = len(self.start)

    def rows(self, c: int, early: bool = False, cap: int | None = None) -> np.ndarray:
        r = np.arange(self.start[c], self.end[c])
        if early:
            m = self.mileage[r]
            keep = m <= m[0] + EARLY_SHARE * (m[-1] - m[0])
            keep[:MIN_SNIPPETS] = True
            r = r[keep]
        if cap and len(r) > cap:
            r = r[np.unique(np.linspace(0, len(r) - 1, cap).round().astype(int))]
        return r


# ---------------------------------------------------------------- car features

def car_row(f: np.ndarray, mileage: np.ndarray) -> list[float]:
    """Car features (BEHAVIOUR + USAGE) from one car's snippet features, ordered by mileage."""
    s = f[:, SN["spread_mv"]]
    n = len(s)
    k = max(1, int(round(0.3 * n)))
    mid = f[:, SN["spread_mid_soc_mv"]]
    mid_med = np.nanmedian(mid) if np.isfinite(mid).any() else np.median(s)
    rho = spearmanr(mileage, s)[0] if np.ptp(s) > 0 and np.ptp(mileage) > 0 else 0.0
    q25, q75 = np.percentile(s, [25, 75])
    log = lambda x: float(np.log(max(x, 1.0)))  # noqa: E731 - spreads in mV, floored at 1 mV
    return [log(np.median(s)), log(np.percentile(f[:, SN["spread_max_mv"]], 95)), log(mid_med), q75 - q25,
            float(np.nan_to_num(rho)), np.median(s[-k:]) - np.median(s[:k]), np.median(f[:, SN["spread_growth_mv"]]),
            np.median(f[:, SN["temp_spread_c"]]), np.percentile(f[:, SN["temp_max_c"]], 95),
            np.percentile(f[:, SN["vmax_rise_mv_per_min"]], 95),
            np.log(n), np.median(mileage), np.ptp(mileage), np.median(f[:, SN["current_abs_a"]]),
            np.median(f[:, SN["soc_start"]]), np.median(f[:, SN["soc_range"]])]


class Table:
    """Car-level features for one view of each car's history (all of it, its first half, or capped)."""

    adjust = False  # mileage-adjusted check: features are residuals of a per-brand linear fit on mileage

    def __init__(self, fl: Fleet, early: bool = False, cap: int | None = None):
        rows, self.spread_max = [], []
        for c in range(fl.n):
            r = fl.rows(c, early, cap)
            rows.append(car_row(fl.F[r], fl.mileage[r]))
            self.spread_max.append(np.sort(fl.F[r, SN["spread_max_mv"]]))
        self.base = np.array(rows, dtype=float)
        self.p95 = np.array([np.percentile(s, 95) for s in self.spread_max])
        self.n_snippets = np.array([len(s) for s in self.spread_max])
        self.brand, self.y, self.car = fl.brand, fl.y, fl.car

    def share(self, cars: np.ndarray, level: float) -> np.ndarray:
        return np.array([1 - np.searchsorted(self.spread_max[i], level, side="right") / len(self.spread_max[i])
                         for i in cars])

    def raw(self, cars: np.ndarray, prep: dict) -> np.ndarray:
        out = np.empty((len(cars), len(CAR_FEATURES)))
        for b in np.unique(self.brand[cars]):
            sel = self.brand[cars] == b
            out[sel] = np.column_stack([self.base[cars[sel]], self.share(cars[sel], prep[int(b)]["level_mv"])])
        return out


def fit_prep(T: Table, cars: np.ndarray) -> dict:
    """Per-brand high-spread level and robust scaling (and, if T.adjust, the linear mileage trend of every
    feature), from the given (training) cars only; no labels are used."""
    prep = {}
    for b in np.unique(T.brand[cars]):
        cb = cars[T.brand[cars] == b]
        level = float(np.median(T.p95[cb]))
        F = np.column_stack([T.base[cb], T.share(cb, level)])
        p = {"level_mv": level}
        if T.adjust:
            mc = T.base[cb, MIL] - T.base[cb, MIL].mean()
            p.update(adj_m=T.base[cb, MIL].mean(), adj_f=F.mean(0),
                     adj_slope=(mc[:, None] * (F - F.mean(0))).sum(0) / max(float((mc * mc).sum()), 1e-12))
            F = _residual(F, T.base[cb, MIL], p)
        q25, med, q75 = np.percentile(F, [25, 50, 75], axis=0)
        sd = F.std(0)
        p.update(median=med, scale=np.where(q75 - q25 > 1e-9, q75 - q25, np.where(sd > 1e-9, sd, 1.0)))
        prep[int(b)] = p
    return prep


def _residual(F: np.ndarray, mileage: np.ndarray, p: dict) -> np.ndarray:
    return F - (p["adj_f"] + np.outer(mileage - p["adj_m"], p["adj_slope"]))


def design(T: Table, cars: np.ndarray, prep: dict, fallback: int | None = None) -> np.ndarray:
    """Scaled CAR_FEATURES, using each car's own brand's prep (or the fallback brand's if it has none)."""
    p = {int(b): prep.get(int(b), prep.get(fallback)) for b in np.unique(T.brand[cars])}
    X = T.raw(cars, p)
    out = np.empty_like(X)
    for b, pb in p.items():
        sel = T.brand[cars] == b
        Xb = _residual(X[sel], T.base[cars[sel], MIL], pb) if "adj_slope" in pb else X[sel]
        out[sel] = np.clip((Xb - pb["median"]) / pb["scale"], -CLIP, CLIP)
    return out


# ---------------------------------------------------------------- models

class OneFeature:
    """Baseline score: one (scaled) feature as it is, nothing fitted."""

    def fit(self, X, y, sample_weight=None):
        return self

    def predict_proba(self, X):
        return np.column_stack([-X[:, 0], X[:, 0]])


def make(name: str, seed: int):
    if name in ("logistic", "spread_only_logistic", "usage_only_logistic"):
        return LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000)
    if name == "boosting":
        return GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.05, subsample=0.8,
                                          min_samples_leaf=3, random_state=seed)
    return OneFeature()


def cols(name: str) -> list[int]:
    return [CAR_FEATURES.index(f) for f in MODELS[name]]


def fit(name: str, seed: int, X: np.ndarray, y: np.ndarray):
    m = make(name, seed)
    if name == "boosting":
        return m.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    return m.fit(X, y)


def score(m, X: np.ndarray) -> np.ndarray:
    return m.predict_proba(X)[:, 1]


def strata(T: Table, cars: np.ndarray) -> np.ndarray:
    return T.brand[cars] * 2 + T.y[cars]


def fit_and_threshold(name: str, T: Table, train: np.ndarray, seed: int):
    """Model on the training cars, plus the flag threshold from inner out-of-fold scores of those cars."""
    c = cols(name)
    inner = np.full(len(train), np.nan)
    for a, b in StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=seed).split(train, strata(T, train)):
        prep = fit_prep(T, train[a])
        m = fit(name, seed, design(T, train[a], prep)[:, c], T.y[train[a]])
        inner[b] = score(m, design(T, train[b], prep)[:, c])
    prep = fit_prep(T, train)
    m = fit(name, seed, design(T, train, prep)[:, c], T.y[train])
    return m, prep, float(np.quantile(inner[T.y[train] == 0], 1 - BUDGET))


def cross_validate(name: str, T_fit: Table, cars: np.ndarray, repeats: int, T_eval: Table | None = None):
    """Out-of-fold scores and flags per repeat; T_eval describes the held-out cars (e.g. first half only)."""
    T_eval = T_eval or T_fit
    S = np.full((repeats, len(cars)), np.nan)
    flag = np.zeros((repeats, len(cars)), bool)
    c = cols(name)
    for r in range(repeats):
        for a, b in StratifiedKFold(FOLDS, shuffle=True, random_state=SEED + r).split(cars, strata(T_fit, cars)):
            m, prep, thr = fit_and_threshold(name, T_fit, cars[a], SEED + r)
            S[r, b] = score(m, design(T_eval, cars[b], prep)[:, c])
            flag[r, b] = S[r, b] >= thr
    return S, flag


def transfer(name: str, T: Table, train: np.ndarray, test: np.ndarray, adapt: bool):
    """Train on all cars of one brand, score the other brand."""
    m, prep, thr = fit_and_threshold(name, T, train, SEED)
    if adapt:  # the target brand is scaled by its own cars (no labels used)
        prep = {**prep, **fit_prep(T, test)}
    s = score(m, design(T, test, prep, fallback=int(T.brand[train][0]))[:, cols(name)])
    return s[None, :], (s >= thr)[None, :]


# ---------------------------------------------------------------- metrics

def auc(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """ROC-AUC along the last axis from ranks (ties count half); y and s broadcast."""
    r = rankdata(s, axis=-1)
    n1 = y.sum(-1)
    n0 = y.shape[-1] - n1
    return ((r * y).sum(-1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def ci(point, boot) -> dict:
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return {"value": r4(point), "ci95": [r4(lo), r4(hi)]}


def summarise(y: np.ndarray, S: np.ndarray, flag: np.ndarray, seed: int = SEED) -> dict:
    """ROC-AUC (mean over repeats) and flag rates, with car-bootstrap 95% CIs (stratified by label)."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    idx = np.concatenate([rng.choice(pos, (BOOT, len(pos))), rng.choice(neg, (BOOT, len(neg)))], axis=1)
    yb = y[idx]
    per_repeat = auc(y[None, :], S)
    boot = np.mean([auc(yb, s[idx]) for s in S], axis=0)
    f = flag.mean(0)

    def rates(ff, yy):
        tp, fp = (ff * yy).sum(-1), (ff * (1 - yy)).sum(-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return tp / yy.sum(-1), np.where(tp + fp > 0, tp / (tp + fp), np.nan), fp / (1 - yy).sum(-1)

    out = {"cars": int(len(y)), "faulty": int(y.sum()), "roc_auc": ci(per_repeat.mean(), boot)}
    if len(S) > 1:
        out["roc_auc"]["per_repeat_min_max"] = [r4(per_repeat.min()), r4(per_repeat.max())]
    mean_score = S.mean(0)
    out["roc_auc"]["p_one_sided_mann_whitney"] = r4(mannwhitneyu(mean_score[pos], mean_score[neg],
                                                                 alternative="greater").pvalue)
    for key, p, b in zip(("recall", "precision", "false_flag_rate"), rates(f, y), rates(f[idx], yb)):
        out[key] = ci(p, b)
    out["expected_flagged"] = {"faulty": round(float(f[pos].sum()), 2), "normal": round(float(f[neg].sum()), 2)}
    return out


def run(models, T_fit: Table, cars: np.ndarray, repeats: int, T_eval: Table | None = None,
        keep: dict | None = None, tag: str = "") -> dict:
    out = {}
    for name in models:
        t = time.time()
        S, flag = cross_validate(name, T_fit, cars, repeats, T_eval)
        out[name] = summarise(T_fit.y[cars], S, flag)
        if keep is not None:
            keep[(tag, name)] = (cars, S, flag)
        a = out[name]["roc_auc"]
        print(f"  {tag:28s} {name:20s} AUC {a['value']:.3f} [{a['ci95'][0]:.3f}, {a['ci95'][1]:.3f}]  "
              f"recall {out[name]['recall']['value']:.2f}  false-flag {out[name]['false_flag_rate']['value']:.2f}"
              f"  ({time.time() - t:.0f}s)")
    return out


def univariate(T: Table, cars: np.ndarray) -> dict:
    """Descriptive ROC-AUC of each raw car feature (higher = faulty), all cars of a brand; not held out."""
    X = T.raw(cars, fit_prep(T, cars))
    y = T.y[cars]
    return {f: r4(auc(y, X[:, j])) for j, f in enumerate(CAR_FEATURES)}


def score_vs_history(T: Table, cars: np.ndarray, S: np.ndarray) -> dict:
    """Spearman correlation of the out-of-fold score with the amount of history, within each label."""
    s, n = S.mean(0), T.n_snippets[cars]
    span = T.base[cars, CAR_FEATURES.index("mileage_span")]
    out = {}
    for lab, name in ((0, "normal"), (1, "faulty")):
        m = T.y[cars] == lab
        out[name] = {"vs_snippets": r4(spearmanr(s[m], n[m])[0]), "vs_mileage_span": r4(spearmanr(s[m], span[m])[0])}
    return out


def examples(fl: Fleet, T: Table, cars: np.ndarray, S: np.ndarray, flag: np.ndarray) -> list[dict]:
    """A typical faulty and a typical normal car (median out-of-fold score within its label)."""
    s, f = S.mean(0), flag.mean(0)
    level = fit_prep(T, cars)[int(T.brand[cars[0]])]["level_mv"]
    out = []
    for lab, kind in ((1, "faulty_typical"), (0, "normal_typical")):
        pick = np.flatnonzero(T.y[cars] == lab)
        i = pick[np.argsort(s[pick])[len(pick) // 2]]
        c = cars[i]
        r = fl.rows(c)
        sel = r[np.unique(np.linspace(0, len(r) - 1, min(EXAMPLE_POINTS, len(r))).round().astype(int))]
        rank = int((s > s[i]).sum()) + 1
        out.append({"kind": kind, "brand": int(T.brand[c]), "car": int(T.car[c]), "label": int(T.y[c]),
                    "snippets": int(len(r)), "points_shown": int(len(sel)),
                    "oof_score_mean": r4(s[i]), "share_of_repeats_flagged": r4(f[i]),
                    "score_rank_among_cars": f"{rank} of {len(cars)}",
                    "high_spread_level_mv": r4(level),
                    "mileage": [round(float(v), 2) for v in fl.mileage[sel]],
                    "spread_mv": [round(float(v), 2) for v in fl.F[sel, SN["spread_mv"]]],
                    "spread_max_mv": [round(float(v), 2) for v in fl.F[sel, SN["spread_max_mv"]]]})
    return out


def data_summary(fl: Fleet, T: Table, checks: dict) -> dict:
    out = {}
    for b in np.unique(fl.brand):
        cb = np.flatnonzero(fl.brand == b)
        n = T.n_snippets[cb]
        lab = fl.labels[fl.labels.brand == b]
        span = T.base[cb, CAR_FEATURES.index("mileage_span")]
        med = T.base[cb, CAR_FEATURES.index("mileage_median")]
        y = fl.y[cb]
        out[f"brand{int(b)}"] = {
            "cars_in_label_file": int(len(lab)), "faulty_in_label_file": int(lab.label.sum()),
            "label_files": lab.label_file.iloc[0],
            "cars_used": int(len(cb)), "faulty_cars_used": int(y.sum()), "snippets_used": int(n.sum()),
            "snippets_converted": checks[int(b)]["snippets"],
            "snippets_per_car_median": {"normal": float(np.median(n[y == 0])), "faulty": float(np.median(n[y == 1]))},
            "mileage_median_of_cars": {"normal": r4(np.median(med[y == 0])), "faulty": r4(np.median(med[y == 1]))},
            "mileage_span_median": {"normal": r4(np.median(span[y == 0])), "faulty": r4(np.median(span[y == 1]))},
            "cars_without_snippets": sorted(set(lab.car) - set(fl.meta.car[fl.meta.brand == b])
                                            - {int(k.split(":")[1]) for k in fl.too_short if k.startswith(f"{b}:")}),
        }
    return out


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path.home() / "evdata" / "vehicle"), help="folder with brandN.npz")
    ap.add_argument("--raw", default=str(Path.home() / "evdata"), help="folder with battery_brandN/label/*.csv")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    args = ap.parse_args()
    warnings.filterwarnings("ignore", message="The least populated class")
    t0 = time.time()
    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    data, raw = Path(args.data).expanduser(), Path(args.raw).expanduser()
    brands = [b for b in BRANDS if (data / f"brand{b}.npz").exists()]
    parts, checks = [], {}
    for b in brands:
        meta, F, chk = load_brand(data, b)
        parts.append((b, meta, F, load_labels(raw, b)))
        checks[b] = chk
    fl = Fleet(parts)
    full, early, capped = Table(fl), Table(fl, early=True), Table(fl, cap=CAP)
    print(f"brands {brands}: {fl.n} cars ({fl.y.sum()} faulty), {len(fl.F):,} snippets; "
          f"left out (< {MIN_SNIPPETS} snippets): {fl.too_short}; snippet labels agree: {fl.snippet_label_agrees:.4f}")
    R = args.repeats
    keep: dict = {}
    by_brand = {b: np.flatnonzero(fl.brand == b) for b in brands}
    allcars = np.arange(fl.n)
    ev: dict = {}
    for b in brands:
        print(f"brand {b} cross-validation ({R} x {FOLDS}-fold over cars)")
        ev[f"brand{b}_cv"] = run(MODELS, full, by_brand[b], R, keep=keep, tag=f"brand{b}")
    if len(brands) > 1:
        print("both brands pooled")
        ev["pooled_cv"] = run(MODELS, full, allcars, R, keep=keep, tag="pooled")
        ev["cross_brand"] = {}
        for a, b in ((3, 2), (2, 3)):
            entry = {}
            for adapt, key in ((False, "strict_training_brand_scaling"), (True, "target_brand_label_free_scaling")):
                entry[key] = {}
                for name in ("logistic", "boosting", "spread_median_only"):
                    S, flag = transfer(name, full, by_brand[a], by_brand[b], adapt)
                    entry[key][name] = summarise(full.y[by_brand[b]], S, flag)
                    v = entry[key][name]["roc_auc"]
                    print(f"  train brand {a} -> test brand {b} {key:34s} {name:20s} AUC {v['value']:.3f} "
                          f"[{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")
            ev["cross_brand"][f"train_brand{a}_test_brand{b}"] = entry

    print("early warning: first half of each car's mileage history")
    ew_models = ["logistic", "boosting", "spread_only_logistic", "spread_median_only"]
    groups = [(f"brand{b}", by_brand[b]) for b in brands] + ([("pooled", allcars)] if len(brands) > 1 else [])
    ew = {"definition": f"held-out cars described only by snippets with mileage <= first + {EARLY_SHARE:.0%} of "
                        f"their mileage range (at least the first {MIN_SNIPPETS} snippets)",
          "snippets_per_car_median": {g: float(np.median(early.n_snippets[c])) for g, c in groups}}
    for g, c in groups:
        ew[g] = {"trained_on_full_histories": run(ew_models, full, c, R, T_eval=early, tag=f"{g} early(full-trained)"),
                 "trained_on_first_halves": run(ew_models, early, c, R, tag=f"{g} early(half-trained)")}
    ev["early_warning"] = ew

    print(f"history-size check: at most {CAP} snippets per car")
    ev["history_capped_check"] = {"definition": f"every car summarised from at most {CAP} snippets evenly spread "
                                                f"over its mileage-ordered history"}
    for g, c in groups:
        ev["history_capped_check"][g] = run(["logistic", "spread_only_logistic", "spread_median_only"], capped, c, R,
                                            tag=f"{g} capped")
    print("mileage-adjusted check: features minus their per-brand linear trend on mileage (training cars)")
    adjusted = copy.copy(full)
    adjusted.adjust = True
    ev["mileage_adjusted_check"] = {
        "definition": "every car feature replaced by its residual from a per-brand linear fit on the car's median "
                      "mileage, fitted on the training cars of each fold (no labels); asks whether the signal "
                      "survives once age (odometer) is accounted for"}
    for g, c in groups:
        ev["mileage_adjusted_check"][g] = run(["logistic", "spread_only_logistic", "spread_median_only"], adjusted, c,
                                              R, tag=f"{g} mileage-adjusted")

    # Final model: the evaluated procedure on every car (pooled when both brands are there).
    final_cars = allcars
    final, prep, thr = fit_and_threshold("logistic", full, final_cars, SEED)
    tag = "pooled" if len(brands) > 1 else f"brand{brands[0]}"
    cars_k, S_k, flag_k = keep[(tag, "logistic")]
    normal_scores = np.sort(S_k.mean(0)[full.y[cars_k] == 0]).astype(np.float32)
    coefs = dict(zip(MODEL_FEATURES, np.round(final.coef_[0], 4).tolist()))
    reference = {}
    for b in brands:
        cb = by_brand[b][full.y[by_brand[b]] == 0]
        X = full.raw(cb, prep)
        reference[int(b)] = {f: [r4(v) for v in np.percentile(X[:, j], [50, 90, 95])]
                             for j, f in enumerate(CAR_FEATURES)}

    uni = {f"brand{b}": univariate(full, by_brand[b]) for b in brands}
    ex_cars, ex_S, ex_flag = keep[(f"brand{3 if 3 in brands else brands[0]}", "logistic")]
    head = {}
    for g, _ in groups:
        cv = ev[f"{g}_cv"]
        head[g] = {"logistic_roc_auc": cv["logistic"]["roc_auc"], "boosting_roc_auc": cv["boosting"]["roc_auc"],
                   "spread_only_logistic_roc_auc": cv["spread_only_logistic"]["roc_auc"],
                   "spread_median_only_roc_auc": cv["spread_median_only"]["roc_auc"],
                   "usage_only_roc_auc": cv["usage_only_logistic"]["roc_auc"],
                   "logistic_at_threshold": {k: cv["logistic"][k] for k in ("recall", "precision", "false_flag_rate")},
                   "early_warning_logistic_roc_auc": {
                       k: ev["early_warning"][g][k]["logistic"]["roc_auc"]
                       for k in ("trained_on_full_histories", "trained_on_first_halves")},
                   "mileage_adjusted_logistic_roc_auc": ev["mileage_adjusted_check"][g]["logistic"]["roc_auc"],
                   "history_capped_logistic_roc_auc": ev["history_capped_check"][g]["logistic"]["roc_auc"]}
    for k, v in ev.get("cross_brand", {}).items():
        head[k] = {s: v[s]["logistic"]["roc_auc"] for s in v}
    metrics = {
        "headline": head,
        "dataset": {
            "name": "EV battery charging snippets, Zhang et al. (Tsinghua/PKU), figshare 10.6084/m9.figshare.23659323",
            "paper": "Zhang et al., Realistic fault detection of Li-ion battery via dynamical deep learning, "
                     "Nature Communications 14:5940 (2023), doi:10.1038/s41467-023-41226-5",
            "brands_used": [f"brand{b}" for b in brands],
            "cars": int(fl.n), "faulty_cars": int(fl.y.sum()), "snippets": int(len(fl.F)),
            "per_brand": data_summary(fl, full, checks),
            "cars_left_out_for_too_little_history": fl.too_short,
            "snippet": "128 consecutive readings, 10 s apart (~21 minutes of one charge); several snippets per charge",
            "label": "1 = vehicle labelled faulty by the dataset authors (per vehicle; the paper says labels come "
                     "from drivers' reports confirmed by engineers, and that data at or near the failure were "
                     "removed). Faults, not fires: nothing here predicts fire.",
            "snippet_label_agrees_with_vehicle_label": r4(fl.snippet_label_agrees),
            "caveat": "Very small samples (9 faulty cars in brand 3, 16 in brand 2). Results are indicative only; "
                      "confidence intervals are wide.",
        },
        "column_mapping": {
            "columns": COLUMNS,
            "how_verified": [
                "column.pkl shipped inside each brand archive lists exactly these names in this order (read with "
                "a restricted unpickler)",
                "the authors' code (DyAD/model/tasks.py, github.com/962086838/Battery_fault_detection_NC_github) "
                "selects inputs by these names via column.pkl",
                "value checks on every reading (checks_per_brand, shares of readings): max cell >= min cell, "
                "max temp >= min temp, cell voltages in the Li-ion range, SOC within 0-100 and non-decreasing "
                "within a snippet, current <= 0 (charging is negative), timestamps 10 s apart",
            ],
            "checks_per_brand": {f"brand{b}": checks[b] for b in brands},
            "not_used": "'volt' is the pack voltage on a brand-specific scale (brand 3 ~10.7x the mean cell "
                        "voltage, brand 2 below it), so no feature uses it",
            "byte_order": "the pickles store little-endian float64; numpy keeps that order on big-endian s390x, "
                          "so values were read correctly (no torch involved)",
        },
        "snippet_features": SNIPPET_FEATURES,
        "car_features": {f: DESCRIPTIONS[f] for f in CAR_FEATURES},
        "model_features": MODEL_FEATURES,
        "usage_features_for_confounder_check": USAGE,
        "models": {
            "logistic": "L2 logistic regression, C=0.1, class_weight balanced, on per-brand robust-scaled features",
            "boosting": "gradient boosting, 100 trees of depth 2, learning rate 0.05, subsample 0.8, "
                        "min 3 cars per leaf, balanced sample weights",
            "spread_only_logistic": "the drift hypothesis on its own: logistic regression (as above) on the "
                                    "cell-voltage spread features only (" + ", ".join(SPREAD_FEATURES) + ")",
            "spread_median_only": "baseline: the car's median cell-voltage spread, nothing fitted",
            "usage_only_logistic": "confounder check: logistic regression on usage features only",
        },
        "evaluation_design": {
            "unit": "car (every metric is per vehicle; all snippets of a car are on the same side of a split)",
            "cross_validation": f"{R} repeats of stratified {FOLDS}-fold over cars (strata brand x label)",
            "fitted_inside_training_folds_only": ["per-brand robust scaling (median/IQR)",
                                                  "high-spread level per brand", "model", "flag threshold"],
            "threshold": f"score reached by {BUDGET:.0%} of normal training cars, from out-of-fold scores of an "
                         f"inner {INNER_FOLDS}-fold split of the training cars",
            "roc_auc": f"mean over repeats of the pooled out-of-fold AUC; 95% CI from {BOOT} car bootstraps "
                       "stratified by label (the same resampled cars in every repeat)",
            "flag_rates": "each car's share of repeats in which it was flagged, averaged over faulty cars (recall) "
                          "or normal cars (false-flag rate); precision = expected flagged faulty / expected flagged",
            "p_one_sided_mann_whitney": "approximate: Mann-Whitney test of repeat-averaged out-of-fold scores",
            "fixed_before_results": "features, model settings, budget, folds and repeats were set before any "
                                    "evaluation was run and were not tuned on the results; the spread-only model "
                                    "and the mileage-adjusted check were added after a first (2-repeat) run to "
                                    "see what drives the signal",
        },
        "evaluation": ev,
        "descriptive_univariate_roc_auc_not_held_out": uni,
        "strongest_single_features_not_held_out": {
            g: dict(sorted(u.items(), key=lambda kv: -abs(kv[1] - 0.5))[:5]) for g, u in uni.items()},
        "score_vs_amount_of_history_spearman": {
            f"brand{b}": score_vs_history(full, *keep[(f"brand{b}", "logistic")][:2]) for b in brands},
        "final_model": {"model": "logistic", "trained_on": tag, "cars": int(len(final_cars)),
                        "threshold": r4(thr), "coefficients_scaled_features": coefs,
                        "high_spread_level_mv": {f"brand{b}": r4(prep[int(b)]["level_mv"]) for b in brands}},
        "examples": examples(fl, full, ex_cars, ex_S, ex_flag),
        "platform": {"machine": platform.machine(), "python": platform.python_version(),
                     "numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "seconds": round(time.time() - t0, 1),
    }
    (ROOT / "results" / "vehicle_metrics.json").write_text(json.dumps(clean(metrics), indent=2, allow_nan=False))
    joblib.dump({"model": final, "features": MODEL_FEATURES, "car_features": CAR_FEATURES,
                 "snippet_features": SNIPPET_FEATURES, "columns": COLUMNS,
                 "prep": {int(b): {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in p.items()}
                          for b, p in prep.items()},
                 "threshold": thr, "budget": BUDGET, "clip": CLIP, "normal_oof_scores": normal_scores,
                 "reference_normal_p50_p90_p95": reference, "brands": brands,
                 "how_to_score": "snippet_features(X) per snippet -> car_row(features ordered by mileage) "
                                 "+ high_spread_share vs prep[brand]['level_mv'] -> (x - median) / scale, "
                                 "clipped to +-clip -> model.predict_proba (see chargeguard.scripts.train_vehicle)",
                 "trained_on": platform.machine(), "sklearn": sklearn.__version__},
                ROOT / "artifacts" / "vehicle.joblib")
    print(f"wrote results/vehicle_metrics.json, artifacts/vehicle.joblib in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
