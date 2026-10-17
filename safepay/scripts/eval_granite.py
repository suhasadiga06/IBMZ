"""Does IBM Granite's second opinion help, and how fast is it?

Evaluation world: seed 6 (fresh), test-third phrases and Civil Comments test
text, i.e. nothing any model was tuned on. Granite sees the same facts the live
service shows it, *without* the relationship model's score or reasons.

Cases (fixed before any Granite output was seen):
  near_miss   not flagged by SafePay, relationship score within 0.2 below the
              case threshold, at least 3 payments: where Granite could rescue
  weak_case   flagged by SafePay with no abusive or threatening words and a
              score under threshold + 0.1: where Granite could dismiss
  control     random abusive and random hard-innocent relationships

Pre-declared rules (not tuned):
  rescue   near_miss becomes a case if Granite p >= 0.5
  dismiss  weak_case is dropped if Granite p < 0.2

Usage: python3 -m safepay.scripts.eval_granite [--dry-run] [--controls 30]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

from safepay.core import granite
from safepay.core.engine import Engine
from safepay.core.policy import Thresholds
from safepay.scripts.train_rel import first_flags, replay
from safepay.sim import world as W

ART = Path(__file__).resolve().parents[2] / "artifacts"
HARD_INNOCENT = {"debt_reminder", "friends", "night_shift_friends", "account_verification", "parent_allowance",
                 "child_support", "couple_or_family"}


def build_cases(controls: int, seed: int) -> tuple[list[dict], dict]:
    text = joblib.load(ART / "text.joblib")
    rel = joblib.load(ART / "rel.joblib")
    t = Thresholds(text.thresholds["harm"], text.thresholds["threat"], rel["threshold"])
    w = W.build("test", 4000, 400, seed=seed)
    eng = Engine(text, rel["model"])
    rep = replay(w, eng)
    flags = first_flags(rep["rows"], t)
    units: dict = {}
    for r in rep["rows"]:
        u = units.setdefault(r["unit"], {"max_rel": 0.0, "msgs": [], "y": r["y"], "content": False})
        u["max_rel"] = max(u["max_rel"], r["rel"])
        if r["ref"]:
            u["msgs"].append(r["ref"])
        u["content"] |= r["harm"] >= t.harm or r["threat"] >= t.threat
    rng = random.Random(seed)
    cases, seen = [], set()

    def add(unit, group):
        if unit in seen:
            return
        seen.add(unit)
        u, st = units[unit], eng.states[unit]
        cases.append({"unit": list(unit), "group": group, "y": u["y"], "kind": rep["kinds"].get(unit, "?"),
                      "max_rel": round(u["max_rel"], 4), "flagged": unit in flags,
                      "facts": granite.facts_from_state(st), "messages": u["msgs"][-12:]})

    for unit, u in units.items():
        st = eng.states[unit]
        if unit not in flags and t.case - 0.2 <= u["max_rel"] < t.case and st.n_tx >= 3:
            add(unit, "near_miss")
        if unit in flags and not u["content"] and u["max_rel"] < t.case + 0.1:
            add(unit, "weak_case")
    for unit, u in units.items():
        if unit in flags and not u["y"]:
            add(unit, "false_alarm")      # informative only: no rule acts on this group
    abusive = [x for x in units if units[x]["y"] and x not in seen]
    innocent = [x for x in units if not units[x]["y"] and rep["kinds"].get(x) in HARD_INNOCENT and x not in seen]
    for x in rng.sample(abusive, min(controls // 2, len(abusive))):
        add(x, "control")
    for x in rng.sample(innocent, min(controls - controls // 2, len(innocent))):
        add(x, "control")
    meta = {"threshold": t.case, "world": {"relationships": len(w.relationships), "payments": len(w.txs)},
            "flagged": len(flags), "abusive": sum(1 for u in units.values() if u["y"]),
            "caught": sum(1 for x in flags if units[x]["y"]), "false_alarms": sum(1 for x in flags if not units[x]["y"])}
    return cases, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--controls", type=int, default=30)
    ap.add_argument("--seed", type=int, default=6, help="6 = development set, 7 = fresh report set")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. v2_seed7")
    args = ap.parse_args()
    cases, meta = build_cases(args.controls, args.seed)
    meta["seed"] = args.seed
    print("world:", meta, flush=True)
    print("cases:", Counter((c["group"], c["y"]) for c in cases), flush=True)
    if args.dry_run:
        return

    sfx = f"_{args.tag}" if args.tag else ""
    cache_f = ART / f"granite_eval_cache{sfx}.jsonl"
    cache = {}
    if cache_f.exists():
        for line in cache_f.read_text().splitlines():
            d = json.loads(line)
            cache[tuple(d["unit"])] = d
    with cache_f.open("a") as out:
        for i, c in enumerate(cases, 1):
            key = tuple(c["unit"])
            if key in cache:
                c.update(cache[key])
                continue
            t0 = time.time()
            op, timings = granite.second_opinion(c["facts"], c["messages"])
            c.update(p=op["p_concerning"], seconds=round(time.time() - t0, 2),
                     prompt_tokens=timings.get("prompt_n"), prompt_ms=round(timings.get("prompt_ms", 0)))
            out.write(json.dumps({k: c[k] for k in ("unit", "p", "seconds", "prompt_tokens", "prompt_ms")}) + "\n")
            out.flush()
            print(f"{i}/{len(cases)} {c['group']:9s} y={c['y']} kind={c['kind']:22s} rel={c['max_rel']:.2f} "
                  f"granite={c['p']:.2f} {c['seconds']:.1f}s", flush=True)

    y = np.array([c["y"] for c in cases])
    p = np.array([c["p"] for c in cases])
    res = {"meta": meta, "n_cases": len(cases),
           "granite_roc_auc": round(float(roc_auc_score(y, p)), 4) if 0 < y.sum() < len(y) else None,
           "granite_accuracy_at_0.5": round(float(((p >= 0.5) == y).mean()), 4),
           "seconds_per_opinion": {"p50": round(statistics.median(c["seconds"] for c in cases), 1),
                                   "p90": round(float(np.percentile([c["seconds"] for c in cases], 90)), 1)}}
    groups = {}
    for g in ("near_miss", "weak_case", "false_alarm", "control"):
        cs = [c for c in cases if c["group"] == g]
        groups[g] = {"n": len(cs), "abusive": sum(c["y"] for c in cs),
                     "granite_says_abuse_when_abusive": sum(1 for c in cs if c["y"] and c["p"] >= 0.5),
                     "granite_says_abuse_when_innocent": sum(1 for c in cs if not c["y"] and c["p"] >= 0.5)}
    res["groups"] = groups
    rescued = [c for c in cases if c["group"] == "near_miss" and c["p"] >= 0.5]
    dismissed = [c for c in cases if c["group"] == "weak_case" and c["p"] < 0.2]
    res["rules"] = {
        "rescue": {"new_cases": len(rescued), "of_which_abusive": sum(c["y"] for c in rescued)},
        "dismiss": {"dropped_cases": len(dismissed), "of_which_innocent": sum(1 - c["y"] for c in dismissed)},
    }
    caught = meta["caught"] + res["rules"]["rescue"]["of_which_abusive"] - (len(dismissed) - res["rules"]["dismiss"]["of_which_innocent"])
    fa = meta["false_alarms"] + (len(rescued) - res["rules"]["rescue"]["of_which_abusive"]) - res["rules"]["dismiss"]["of_which_innocent"]
    res["whole_world"] = {"safepay_alone": {"caught": meta["caught"], "false_alarms": meta["false_alarms"]},
                          "with_granite_rules": {"caught": caught, "false_alarms": fa}, "abusive": meta["abusive"]}
    res["by_kind"] = {k: round(float(np.mean([c["p"] for c in cases if c["kind"] == k])), 3)
                      for k in sorted({c["kind"] for c in cases})}
    print(json.dumps({k: v for k, v in res.items() if k != "by_kind"}, indent=1), flush=True)
    print("mean Granite p by kind:", res["by_kind"], flush=True)
    res["relationship_model_roc_auc_same_cases"] = round(float(roc_auc_score(y, [c["max_rel"] for c in cases])), 4)
    (ART / f"granite_eval{sfx}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
