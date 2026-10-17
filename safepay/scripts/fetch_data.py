"""Fetch the public text data SafePay trains and tests on. Standard library only.

  google/civil_comments (CC0-1.0): real human comments with crowd-rated
      toxicity, threat, insult... scores. pyarrow has no s390x build, so we
      pull rows as JSON from the Hugging Face datasets API instead of Parquet:
      every comment rated toxic, plus a random sample, from the train and test
      splits separately. Only texts short enough to be a payment reference
      are kept.
  conversationai/unintended-ml-bias-analysis (Apache-2.0): Jigsaw's identity
      template sentences, used to test for unfair false alarms.

Every page is cached, so an interrupted fetch resumes where it stopped.
Usage: python3 -m safepay.scripts.fetch_data
"""
from __future__ import annotations

import csv
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data"
CACHE = DATA / "cache"
API = "https://datasets-server.huggingface.co"
DS = "google/civil_comments"
PAGE = 100
MAX_CHARS = 300          # keep texts a payment reference could plausibly carry
LABELS = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit"]
BIAS_URL = ("https://raw.githubusercontent.com/conversationai/unintended-ml-bias-analysis/"
            "main/sentence_templates/en_sentence_templates.csv")

# split -> (rows in split, random pages to sample)
SPLITS = {"train": (1_804_874, 900), "test": (97_320, 300)}
TOXIC_WHERE = urllib.parse.quote('"toxicity">=0.5')


def _get(url: str, tries: int = 8) -> bytes:
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if attempt == tries - 1 or e.code not in (429, 500, 502, 503, 504):
                raise
            time.sleep(int(e.headers.get("Retry-After") or 0) or min(5 * 2 ** attempt, 90))
        except OSError:
            if attempt == tries - 1:
                raise
            time.sleep(5 * 2 ** attempt)
    raise RuntimeError("unreachable")


def _page(kind: str, split: str, off: int) -> list[dict]:
    f = CACHE / f"{split}_{kind}_{off}.json"
    if not f.exists():
        if kind == "toxic":
            url = (f"{API}/filter?dataset={DS}&config=default&split={split}"
                   f"&where={TOXIC_WHERE}&offset={off}&length={PAGE}")
        else:
            url = f"{API}/rows?dataset={DS}&config=default&split={split}&offset={off}&length={PAGE}"
        body = json.loads(_get(url))
        rows = [r["row"] for r in body["rows"]]
        f.write_text(json.dumps(rows), encoding="utf-8")
        time.sleep(0.4)
    return json.loads(f.read_text(encoding="utf-8"))


def fetch_split(split: str) -> Path:
    total, n_random = SPLITS[split]
    first = json.loads(_get(f"{API}/filter?dataset={DS}&config=default&split={split}"
                            f"&where={TOXIC_WHERE}&offset=0&length=1"))
    n_toxic = first["num_rows_total"]
    jobs = [("toxic", off) for off in range(0, n_toxic, PAGE)]
    rng = random.Random(7 if split == "train" else 8)
    jobs += [("random", off) for off in sorted(rng.sample(range(0, total - PAGE, PAGE), n_random))]
    print(f"{split}: {n_toxic} toxic rows + {n_random * PAGE} random rows, {len(jobs)} pages", flush=True)

    seen: set[str] = set()
    kept: list[dict] = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for i, rows in enumerate(pool.map(lambda j: _page(j[0], split, j[1]), jobs), 1):
            for r in rows:
                text = (r.get("text") or "").strip()
                if not text or len(text) > MAX_CHARS or text in seen:
                    continue
                seen.add(text)
                kept.append({"text": text, **{k: round(float(r.get(k) or 0.0), 4) for k in LABELS}})
            if i % 100 == 0:
                print(f"  {split}: {i}/{len(jobs)} pages, {len(kept)} short texts kept", flush=True)

    out = DATA / f"cc_{split}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", *LABELS])
        w.writeheader()
        w.writerows(kept)
    tox = sum(r["toxicity"] >= 0.5 for r in kept)
    thr = sum(r["threat"] >= 0.5 for r in kept)
    print(f"wrote {out.name}: {len(kept)} rows, {tox} toxic, {thr} threats", flush=True)
    return out


def fetch_bias() -> None:
    out = DATA / "bias_templates_en.csv"
    if not out.exists():
        out.write_bytes(_get(BIAS_URL))
    print(f"wrote {out.name}", flush=True)


if __name__ == "__main__":
    CACHE.mkdir(parents=True, exist_ok=True)
    fetch_bias()
    fetch_split("test")
    fetch_split("train")
