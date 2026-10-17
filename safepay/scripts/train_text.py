"""Train and evaluate Model 1 (message harm + threat).

  train      Civil Comments train split (all toxic/borderline + random rest)
  threshold  tuned on the validation split
  score      once on the test split, overall and for payment-length texts
  robustness the same toxic test texts after evasion tricks (l33t, d.o.t.s,
             s p a c e s, look-alike letters, emoji), with and without our
             canonicaliser, to show what it buys
  fairness   Jigsaw identity templates: false alarms on non-toxic sentences
             that mention a group ("i am a gay man", "proud muslim")

Usage: python3 -m safepay.scripts.train_text [--max-train 260000]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)

from safepay.core.normalize import EMOJI_WORDS
from safepay.core.textmodel import MessageModel

ROOT = Path(__file__).resolve().parents[2]
DATA, ART = ROOT / "data", ROOT / "artifacts"
LABEL_CUT = {"harm": ("toxicity", 0.5), "threat": ("threat", 0.5)}
IDENTITY = ["lesbian", "gay", "bisexual", "transgender", "trans", "queer", "lgbt", "lgbtq", "homosexual",
            "straight", "heterosexual", "nonbinary", "african american", "african", "black", "white",
            "european", "hispanic", "latino", "latina", "latinx", "mexican", "asian", "indian",
            "middle eastern", "chinese", "japanese", "christian", "muslim", "jewish", "buddhist",
            "catholic", "protestant", "sikh", "hindu", "old", "elderly", "young", "teenage", "blind",
            "deaf", "paralyzed", "female", "male", "woman", "man"]


def labels(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {k: (df[col] >= cut).astype(int).values for k, (col, cut) in LABEL_CUT.items()}


def best_threshold(y, p) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return float(thr[np.argmax(f1[:-1])])


def report(y, p, thr) -> dict:
    pred = (p >= thr).astype(int)
    fp = int(((pred == 1) & (y == 0)).sum())
    return {"n": int(len(y)), "positives": int(y.sum()),
            "precision": round(precision_score(y, pred, zero_division=0), 4),
            "recall": round(recall_score(y, pred, zero_division=0), 4),
            "f1": round(f1_score(y, pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y, p), 4) if 0 < y.sum() < len(y) else None,
            "false_positive_rate": round(fp / max(int((y == 0).sum()), 1), 4)}


# -- evasion tricks, applied to real toxic test texts ---------------------------
HOMO = {"a": "а", "e": "е", "o": "о", "c": "с", "p": "р", "x": "х", "y": "у", "i": "і"}
LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}
WORD_EMOJI = {"kill": "🔪", "dead": "💀", "die": "⚰", "shoot": "🔫", "gun": "🔫", "bomb": "💣", "pig": "🐷"}


def perturb(text: str, rng: random.Random) -> str:
    words = text.split()
    if not words:
        return text
    trick = rng.choice(["leet", "dots", "spaces", "homoglyph", "emoji"])
    idx = sorted(rng.sample(range(len(words)), k=max(1, len(words) // 3)))
    for i in idx:
        w = words[i]
        if len(w) < 3 or not w.isalpha():
            continue
        if trick == "leet":
            words[i] = "".join(LEET.get(c.lower(), c) for c in w)
        elif trick == "dots":
            words[i] = ".".join(w)
        elif trick == "spaces" and len(w) >= 4:
            words[i] = " ".join(w)
        elif trick == "homoglyph":
            words[i] = "".join(HOMO.get(c, c) for c in w)
        elif trick == "emoji" and w.lower() in WORD_EMOJI:
            words[i] = WORD_EMOJI[w.lower()]
    return " ".join(words)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train", type=int, default=260_000)
    args = ap.parse_args()
    ART.mkdir(exist_ok=True)

    train = pd.read_csv(DATA / "cc_train.csv").dropna(subset=["text"])
    val = pd.read_csv(DATA / "cc_val.csv").dropna(subset=["text"])
    test = pd.read_csv(DATA / "cc_test.csv").dropna(subset=["text"])
    if len(train) > args.max_train:
        hard = train[(train.toxicity >= 0.5) | (train.threat >= 0.3)]
        rest = train.drop(hard.index).sample(n=max(args.max_train - len(hard), 0), random_state=0)
        train = pd.concat([hard, rest])
    print(f"train {len(train)} | val {len(val)} | test {len(test)}", flush=True)

    t0 = time.time()
    model = MessageModel().fit(train.text.values, labels(train))
    fit_s = time.time() - t0
    print(f"fit {fit_s:.0f}s", flush=True)

    pv = model.proba(val.text.values)
    yv = labels(val)
    model.thresholds = {k: best_threshold(yv[k], pv[k]) for k in model.heads}

    t0 = time.time()
    pt = model.proba(test.text.values)
    ms = 1000 * (time.time() - t0) / len(test)
    yt = labels(test)
    short = (test.text.str.len() <= 140).values
    out = {"fit_seconds": round(fit_s, 1), "ms_per_message_batch": round(ms, 3),
           "thresholds": {k: round(v, 4) for k, v in model.thresholds.items()},
           "rows": {"train": len(train), "val": len(val), "test": len(test), "test_short": int(short.sum())}}
    for k in model.heads:
        out[k] = {"test_all": report(yt[k], pt[k], model.thresholds[k]),
                  "test_payment_length": report(yt[k][short], pt[k][short], model.thresholds[k])}
        print(k, out[k], flush=True)

    # Robustness: toxic test texts, perturbed. Recall with and without canonicalising.
    rng = random.Random(1)
    tox = test[test.toxicity >= 0.5].text.tolist()
    pert = [perturb(t, rng) for t in tox]
    with_c = model.proba(pert)["harm"] >= model.thresholds["harm"]
    raw = model._X_raw(pert)
    without_c = model.heads["harm"].predict_proba(raw)[:, 1] >= model.thresholds["harm"]
    clean = model.proba(tox)["harm"] >= model.thresholds["harm"]
    out["robustness"] = {"n": len(tox), "recall_clean": round(float(clean.mean()), 4),
                         "recall_evasion_with_canonicaliser": round(float(with_c.mean()), 4),
                         "recall_evasion_without_canonicaliser": round(float(without_c.mean()), 4)}
    print("robustness", out["robustness"], flush=True)

    # Fairness: Jigsaw identity templates.
    bias = pd.read_csv(DATA / "bias_templates_en.csv")
    bias["y"] = (bias.toxicity == "toxic").astype(int)
    pb = model.proba(bias.phrase.values)["harm"]
    flag = pb >= model.thresholds["harm"]
    non = bias.y.values == 0
    per_term = {}
    for term in IDENTITY:
        m = bias.phrase.str.contains(rf"\b{re.escape(term)}\b", case=False, regex=True).values & non
        if m.sum() >= 20:
            per_term[term] = round(float(flag[m].mean()), 4)
    worst = sorted(per_term.items(), key=lambda kv: -kv[1])[:8]
    out["fairness"] = {"templates": int(len(bias)), "nontoxic": int(non.sum()),
                       "fpr_all_nontoxic": round(float(flag[non].mean()), 4),
                       "recall_toxic_templates": round(float(flag[~non].mean()), 4),
                       "auc": round(roc_auc_score(bias.y.values, pb), 4),
                       "worst_terms_fpr": dict(worst), "per_term_fpr": per_term}
    print("fairness", {k: v for k, v in out["fairness"].items() if k != "per_term_fpr"}, flush=True)

    joblib.dump(model, ART / "text.joblib", compress=3)
    (ART / "metrics_text.json").write_text(json.dumps(out, indent=2))
    print("saved", flush=True)


if __name__ == "__main__":
    main()
