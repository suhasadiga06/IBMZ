"""How fast ChargeGuard screens a meter reading on this machine (run it on the LinuxONE VM).

  tree formats   risk-model inference with Snap ML on the CPU ("compress_trees") and on the
                 IBM z16/z17 on-chip AI accelerator ("zdnn_tensors"): one reading at a time
                 (the live case) and in batches; predictions are checked to agree.
  pipeline       the full per-reading path in-process: features -> model -> ladder -> reasons ->
                 SQLite, over recorded readings from the held-out stations.
  ocpp           the round trip over OCPP WebSockets, from results/replay_latency.json if a
                 replay has been run.

    python -m chargeguard.scripts.benchmark
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np

from chargeguard.core import model as M
from chargeguard.core.features import COLS, FEATURES, Reading
from chargeguard.core.screener import Screener
from chargeguard.core.store import Store
from chargeguard.core.tariff import Tariff
from chargeguard.scripts import train as T

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "risk.joblib"


def machine() -> dict:
    out = {"machine": platform.machine(), "python": platform.python_version()}
    try:
        lscpu = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=5).stdout
        for line in lscpu.splitlines():
            k, _, v = line.partition(":")
            if k.strip() in ("Machine type", "CPU(s)", "Model name"):
                out[k.strip()] = v.strip()
        out["ibm_z"] = {"9175": "IBM z17", "3931": "IBM z16", "3932": "IBM z16 A02/AGZ"}.get(out.get("Machine type"), "")
    except (OSError, subprocess.SubprocessError):
        pass
    return out


def time_single(model, rows: np.ndarray) -> float:
    for r in rows[:50]:
        M.proba(model, r[None, :])
    t0 = time.perf_counter()
    for r in rows:
        M.proba(model, r[None, :])
    return (time.perf_counter() - t0) / len(rows) * 1e6


def granite_notes() -> dict:
    """Two incident notes back to back: the second reuses the cached instruction prefix."""
    from chargeguard.core import granite
    if not granite.available():
        return {"error": "Granite server not reachable at " + granite.URL}
    first = {"cp": "SZ-03", "connector": 2, "battery": "LFP", "minute": 4, "percentile": 0.9997,
             "action": "set_charging_profile", "current_limit_amps": 60.0, "gun_temperatures_c": "40 and 41",
             "energy_kwh": 3.0, "reasons": ["the two gun temperature sensors read unusually close together "
                                            "(at most 1 °C apart); 99% of normal charges stay above 3 °C"]}
    second = dict(first, cp="SZ-10", connector=4, battery="NMC", minute=6, percentile=0.995,
                  gun_temperatures_c="53 and 55", energy_kwh=4.4,
                  reasons=["gun temperature (sensor 1) is now 53 °C; 95% of normal charges stay under 48 °C"])
    runs = []
    for f in (first, second):
        t0 = time.perf_counter()
        note, tm = granite.write_timed(f, timeout=900)
        runs.append({"seconds": round(time.perf_counter() - t0, 1), "prompt_tokens_processed": tm.get("prompt_n"),
                     "prompt_seconds": round(tm.get("prompt_ms", 0) / 1000, 1),
                     "tokens_generated": tm.get("predicted_n"),
                     "tokens_per_second": round(tm.get("predicted_per_second", 0), 2), "note": note})
        print("granite", {k: v for k, v in runs[-1].items() if k != "note"}, "\n ", note, flush=True)
    return {"model": "granite-3.3-2b-instruct Q4_K_M (big-endian), llama.cpp, shared with SafePay",
            "max_tokens": granite.MAX_TOKENS, "runs": runs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--granite", action="store_true", help="also time two Granite incident notes")
    args = ap.parse_args()
    # Granite first, before this process loads 1.4M readings: on an 8 GB VM that data pushes
    # parts of the model file out of memory and notes slow down several-fold.
    notes = granite_notes() if args.granite else None
    bundle = joblib.load(ART)
    cols = [FEATURES.index(f) for f in bundle["features"]]
    df = T.load(ROOT / "data" / "ev30.csv.gz")
    X, _ = T.cached_features(df, ROOT / "data" / "ev30.csv.gz")
    test_rows = np.flatnonzero(df["class_judge"].isin(bundle["test_stations"]).to_numpy())
    rng = np.random.default_rng(0)
    Xs = X[rng.choice(test_rows, 4096, replace=False)][:, cols].astype(np.float32)
    out = {"platform": machine(), "engine": bundle["engine"], "features": len(cols), "tree_formats": {}}

    reference = M.proba(joblib.load(ART)["model"], Xs[:512])
    formats = ("compress_trees", "zdnn_tensors") if bundle["engine"] == "snapml" else ("sklearn",)
    for fmt in formats:
        model = joblib.load(ART)["model"]  # Snap ML can't re-optimise an optimised model: start fresh
        try:
            got = M.optimise(model, Xs, fmt) if fmt != "sklearn" else "sklearn"
        except Exception as e:  # e.g. no accelerator on this machine
            out["tree_formats"][fmt] = {"error": str(e)[:200]}
            continue
        single = time_single(model, Xs[:1000])
        batch = {}
        for n in (32, 256, 4096):
            t0 = time.perf_counter()
            reps = max(1, 8192 // n)
            for _ in range(reps):
                M.proba(model, Xs[:n])
            batch[str(n)] = round((time.perf_counter() - t0) / (reps * n) * 1e6, 3)
        diff = float(np.max(np.abs(M.proba(model, Xs[:512]) - reference)))
        out["tree_formats"][fmt] = {"reported": got, "single_reading_us": round(single, 2),
                                    "batch_us_per_reading": batch, "max_abs_diff_vs_unoptimised": round(diff, 6)}
        print(fmt, out["tree_formats"][fmt], flush=True)

    ok = {k: v for k, v in out["tree_formats"].items() if "single_reading_us" in v}
    if ok:
        out["fastest_single_reading"] = min(ok, key=lambda k: ok[k]["single_reading_us"])

    # Full in-process pipeline over recorded readings from the held-out stations.
    with tempfile.TemporaryDirectory() as d:
        sc = Screener(ART, Store(Path(d) / "bench.db"), Tariff(ROOT / "artifacts" / "tariff.json"), notes=False)
        sub = df[df["class_judge"].isin(bundle["test_stations"])]
        txs = sub["transaction_id"].unique()[:120]
        sub = sub[sub["transaction_id"].isin(txs)]
        lat, n = [], 0
        for i, (txid, g) in enumerate(sub.groupby("transaction_id", sort=False)):
            sc.battery_reported("BENCH", 1, int(g["types"].iloc[0]))
            sc.start("BENCH", 1, i, f"card-{txid}", "2021-01-01T10:00:00+08:00", 0.0, True)
            for row in g.itertuples(index=False):
                r = Reading(**{k: float(getattr(row, c)) for k, c in COLS.items()})
                t0 = time.perf_counter()
                sc.reading(i, "2021-01-01T10:00:00+08:00", r)
                lat.append((time.perf_counter() - t0) * 1e6)
                n += 1
            sc.end(i, "2021-01-01T11:00:00+08:00", "Local")
        lat = np.array(lat[200:])
        out["pipeline"] = {"readings": n, "tree_format": sc.tree_format,
                           "us_p50": round(float(np.percentile(lat, 50)), 1),
                           "us_p99": round(float(np.percentile(lat, 99)), 1),
                           "readings_per_second_one_core": round(1e6 / float(np.mean(lat)))}
        print("pipeline", out["pipeline"], flush=True)

    rep = ROOT / "results" / "replay_latency.json"
    if rep.exists():
        out["ocpp_round_trip"] = json.loads(rep.read_text())
    if notes is not None:
        out["granite_notes"] = notes
    elif (ROOT / "results" / "benchmark.json").exists():  # keep an earlier Granite measurement
        old = json.loads((ROOT / "results" / "benchmark.json").read_text())
        if "granite_notes" in old:
            out["granite_notes"] = old["granite_notes"]
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "benchmark.json").write_text(json.dumps(out, indent=2))
    print("wrote results/benchmark.json")


if __name__ == "__main__":
    main()
