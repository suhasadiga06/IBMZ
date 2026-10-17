"""Fetch the two public prompt-injection datasets as CSV.

Standard library only. pyarrow has no s390x wheels, so instead of reading
Parquet we pull rows as JSON from the Hugging Face datasets API, sampled at
random offsets with a fixed seed so the training set is reproducible.

  source A: guychuk/benign-malicious-prompt-classification (Apache-2.0) -> train.csv
  source B: jackhhao/jailbreak-classification (Apache-2.0), different upstream
            collections -> source_b_calib.csv (its train split) and
            source_b_test.csv (its test split, never used for fitting or tuning)

Usage: python3 -m agentguard.scripts.fetch_data [--rows 60000] [--from-cache]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data"
ROWS_API = ("https://datasets-server.huggingface.co/rows?dataset={ds}"
            "&config=default&split=train&offset={off}&length={n}")
TRAIN_DS = "guychuk/benign-malicious-prompt-classification"
TRAIN_TOTAL = 464_470
TEST_CSV = ("https://huggingface.co/datasets/jackhhao/jailbreak-classification"
            "/resolve/main/default/jailbreak_dataset_{split}.csv")
PAGE = 100


def _get(url: str, tries: int = 8) -> bytes:
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if attempt == tries - 1 or e.code not in (429, 500, 502, 503, 504):
                raise
            wait = int(e.headers.get("Retry-After") or 0) or min(5 * 2 ** attempt, 90)
            time.sleep(wait)
        except OSError:
            if attempt == tries - 1:
                raise
            time.sleep(5 * 2 ** attempt)
    raise RuntimeError("unreachable")


def fetch_train(n_rows: int, seed: int = 42) -> Path:
    pages = n_rows // PAGE
    rng = random.Random(seed)
    offsets = sorted(rng.sample(range(0, TRAIN_TOTAL - PAGE, PAGE), pages))

    # One cached JSON file per page, so an interrupted fetch resumes.
    cache = DATA / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    def page(off: int) -> list[dict]:
        f = cache / f"{off}.json"
        if not f.exists():
            body = json.loads(_get(ROWS_API.format(ds=TRAIN_DS, off=off, n=PAGE)))
            rows = [{"text": r["row"]["prompt"], "label": int(r["row"]["label"])} for r in body["rows"]]
            f.write_text(json.dumps(rows), encoding="utf-8")
            time.sleep(0.5)  # stay under the API's rate limit
        return json.loads(f.read_text(encoding="utf-8"))

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i, chunk in enumerate(pool.map(page, offsets), 1):
            rows.extend(chunk)
            if i % 50 == 0:
                print(f"  {i}/{pages} pages", flush=True)
    out = DATA / "train.csv"
    _write(out, rows)
    return out


def fetch_test() -> None:
    """Second source, kept in its own two files: one for calibration, one never touched."""
    for split, name in (("train", "source_b_calib.csv"), ("test", "source_b_test.csv")):
        text = _get(TEST_CSV.format(split=split)).decode("utf-8")
        reader = csv.DictReader(io.StringIO(text))
        _write(DATA / name, [{"text": r["prompt"], "label": int(r["type"] == "jailbreak")} for r in reader])


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label"])
        w.writeheader()
        w.writerows(rows)
    pos = sum(r["label"] for r in rows)
    print(f"wrote {path.name}: {len(rows)} rows, {pos} malicious ({pos / max(len(rows), 1):.1%})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=60_000)
    ap.add_argument("--from-cache", action="store_true",
                    help="build train.csv from pages already downloaded, without fetching more")
    args = ap.parse_args()
    if not (DATA / "source_b_test.csv").exists():
        fetch_test()
    if args.from_cache:
        rows = [r for f in sorted((DATA / "cache").glob("*.json"))
                for r in json.loads(f.read_text(encoding="utf-8"))]
        _write(DATA / "train.csv", rows)
    else:
        fetch_train(args.rows)
