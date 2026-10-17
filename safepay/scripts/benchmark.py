"""Measure SafePay's speed on this machine and write artifacts/benchmark.json.

  pipeline     the full screening path, in process, with SQLite persistence,
               over a fresh synthetic world (unseen seed)
  tree formats relationship-model inference with Snap ML on the CPU
               ("compress_trees") and on the on-chip AI accelerator ("zdnn_tensors")
  http         optional: ISO 20022 XML over HTTP to a running server (--url)
  granite      optional: tokens/s from the local llama.cpp server (--granite)

Usage: python3 -m safepay.scripts.benchmark [--url http://127.0.0.1:8090] [--granite]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
import time
import urllib.request
from pathlib import Path

import joblib
import numpy as np

from safepay.core import granite, iso20022, relmodel
from safepay.core.privacy import Privacy
from safepay.core.screener import Screener
from safepay.core.store import Store
from safepay.sim import world as W

ART = Path(__file__).resolve().parents[2] / "artifacts"


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(int(q * len(xs)), len(xs) - 1)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="")
    ap.add_argument("--granite", action="store_true")
    ap.add_argument("--innocent", type=int, default=3000)
    ap.add_argument("--abusive", type=int, default=300)
    args = ap.parse_args()

    text = joblib.load(ART / "text.joblib")
    bundle = joblib.load(ART / "rel.joblib")
    world = W.build("test", args.innocent, args.abusive, seed=77)
    payments = [iso20022.new_payment(world.customers[t.sender], t.sender, world.customers[t.recipient], t.recipient,
                                     t.amount, t.reference, when=t.ts) for t in world.txs]
    out = {"world": {"relationships": len(world.relationships), "payments": len(payments)}}

    # 1. Relationship model engines.
    rng = np.random.default_rng(0)
    probe = Screener(text, bundle, Store(":memory:"), Privacy(os.urandom(32)))
    X = np.array([probe.engine.observe(p.created, p.debtor_account, p.creditor_account, p.amount, p.reference).features
                  for p in payments[:3000]], dtype=np.float32)
    out["tree_formats"] = {}
    for fmt in ("compress_trees", "zdnn_tensors"):
        # Snap ML cannot switch an already-optimised model to another engine: start fresh each time.
        model = joblib.load(ART / "rel.joblib")["model"]
        try:
            got = relmodel.optimise(model, X, fmt)
        except Exception as e:
            out["tree_formats"][fmt] = {"error": str(e)[:200]}
            continue
        rows = X[rng.integers(0, len(X), 500)]
        for r in rows[:50]:
            model.predict_proba(r[None, :])
        t0 = time.perf_counter()
        for r in rows:
            model.predict_proba(r[None, :])
        single = (time.perf_counter() - t0) / len(rows) * 1e6
        batch = X[:1000]
        t0 = time.perf_counter()
        for _ in range(5):
            model.predict_proba(batch)
        per_row = (time.perf_counter() - t0) / (5 * len(batch)) * 1e6
        out["tree_formats"][fmt] = {"reported": got, "single_us": round(single, 2), "batch_us_per_row": round(per_row, 3)}
        print(fmt, out["tree_formats"][fmt], flush=True)
    fastest = min((k for k in out["tree_formats"] if "single_us" in out["tree_formats"][k]),
                  key=lambda k: out["tree_formats"][k]["single_us"])
    out["fastest_tree_format"] = fastest
    # The pipeline below uses the same engine as the live service (Snap ML "auto").
    bundle = joblib.load(ART / "rel.joblib")
    out["service_tree_format"] = relmodel.optimise(bundle["model"], X, "auto")

    # 2. Full pipeline with persistence.
    with tempfile.TemporaryDirectory() as d:
        sc = Screener(text, bundle, Store(Path(d) / "bench.db"), Privacy(os.urandom(32)))
        for p in payments[:500]:          # warm caches and code paths
            sc.screen(p)
        sc = Screener(text, bundle, Store(Path(d) / "bench2.db"), Privacy(os.urandom(32)))
        lat = []
        t0 = time.perf_counter()
        for p in payments:
            s = time.perf_counter()
            sc.screen(p)
            lat.append((time.perf_counter() - s) * 1000)
        wall = time.perf_counter() - t0
    out["pipeline"] = {"payments": len(payments), "seconds": round(wall, 2),
                       "payments_per_second": round(len(payments) / wall, 1),
                       "latency_ms": {"p50": round(pct(lat, .5), 3), "p95": round(pct(lat, .95), 3),
                                      "p99": round(pct(lat, .99), 3), "mean": round(statistics.mean(lat), 3)}}
    print("pipeline", out["pipeline"], flush=True)

    # 3. HTTP, ISO 20022 XML.
    if args.url:
        lat = []
        sample = payments[:2000]
        t0 = time.perf_counter()
        for p in sample:
            body = iso20022.build_pacs008(p).encode()
            req = urllib.request.Request(args.url + "/v1/screen", data=body, headers={"Content-Type": "application/xml"})
            s = time.perf_counter()
            with urllib.request.urlopen(req, timeout=30) as r:
                r.read()
            lat.append((time.perf_counter() - s) * 1000)
        wall = time.perf_counter() - t0
        out["http"] = {"requests": len(sample), "requests_per_second": round(len(sample) / wall, 1),
                       "latency_ms": {"p50": round(pct(lat, .5), 2), "p99": round(pct(lat, .99), 2)}}
        print("http", out["http"], flush=True)

    # 4. Granite.
    if args.granite and granite.available():
        facts = {"payments_from_sender": 42, "over_days": 9, "share_of_payments_under_one_dollar": "88%",
                 "share_sent_between_midnight_and_5am": "40%", "payments_back_from_recipient": 1}
        msgs = ["unblock me", "answer your phone", "saw you at the gym today", "you can't hide from me"]
        t0 = time.time()
        _, timings = granite.summary(facts, msgs)
        took = time.time() - t0
        out["granite"] = {"model": "granite-3.3-2b-instruct Q4_K_M (big-endian GGUF)", "seconds_per_brief": round(took, 1),
                          "prompt_tokens_per_second": round(timings.get("prompt_per_second", 0), 2),
                          "generate_tokens_per_second": round(timings.get("predicted_per_second", 0), 2),
                          "generated_tokens": timings.get("predicted_n")}
        print("granite", out["granite"], flush=True)

    out["notes"] = ("Single process, no batching across payments. "
                    "Snap ML tree inference is microseconds either way; the message model and SQLite write dominate.")
    (ART / "benchmark.json").write_text(json.dumps(out, indent=2))
    print("saved", flush=True)


if __name__ == "__main__":
    main()
