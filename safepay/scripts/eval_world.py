"""Score a saved model on freshly generated worlds, exactly as the live service would.

Two uses:
  python3 -m safepay.scripts.eval_world --seed 9 --tag before_switch
      one mixed world: catch rate, false alarms and delay per relationship type
  python3 -m safepay.scripts.eval_world --abusive 0 --innocent 25000 --chunks 8 --seed 101 --tag prevalence
      a very large innocent-only world, built in chunks to fit in memory, to
      measure the false-alarm rate at a realistic share of abusers

Writes artifacts/eval_<tag>.json.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import joblib
from scipy.stats import beta

from safepay.core.engine import Engine
from safepay.core.policy import Thresholds
from safepay.scripts.train_rel import evaluate, first_flags, message_outcomes, replay
from safepay.sim import world as W

ART = Path(__file__).resolve().parents[2] / "artifacts"


def fpr_bounds(fa: int, n: int) -> tuple[float, float]:
    """Clopper-Pearson 95% interval for a false-alarm rate."""
    lo = 0.0 if fa == 0 else float(beta.ppf(0.025, fa, n - fa + 1))
    hi = 1.0 if fa == n else float(beta.ppf(0.975, fa + 1, n - fa))
    return lo, hi


def precision_at(recall: float, fpr: float, prevalence: float) -> float:
    tp, fp = recall * prevalence, fpr * (1 - prevalence)
    return tp / (tp + fp) if tp + fp else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ART / "rel.joblib"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--innocent", type=int, default=4000)
    ap.add_argument("--abusive", type=int, default=400)
    ap.add_argument("--chunks", type=int, default=1)
    ap.add_argument("--recall", type=float, default=None, help="recall to use for precision-at-prevalence")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    text = joblib.load(ART / "text.joblib")
    bundle = joblib.load(args.model)
    t = Thresholds(text.thresholds["harm"], text.thresholds["threat"], bundle["threshold"])
    n_feat = getattr(bundle["model"], "n_features_in_", None) or len(bundle.get("features", []))

    totals = {"relationships": 0, "payments": 0, "abusive": 0, "caught": 0, "innocent": 0, "false_alarms": 0,
              "innocent_messages": 0, "innocent_hidden": 0, "innocent_hidden_without_protect": 0,
              "abusive_messages": 0, "abusive_hidden": 0, "innocent_relationships_protect_hidden": 0}
    fa_kinds, n_kinds, caught_kinds = Counter(), Counter(), Counter()
    single = None
    t0 = time.time()
    for c in range(args.chunks):
        w = W.build(args.split, args.innocent, args.abusive, seed=args.seed + c)
        rep = replay(w, Engine.for_bundle(text, bundle))
        units = {r["unit"] for r in rep["rows"]}
        flags = first_flags(rep["rows"], t)
        first_abuse = {(r.sender, r.recipient): r.first_abuse for r in w.relationships if r.label}
        ev = evaluate(flags, units, rep["abusive"], rep["kinds"], first_abuse)
        msg = message_outcomes(rep["rows"], flags, t.harm, t.threat, rep["kinds"])
        msg_np = message_outcomes(rep["rows"], flags, t.harm, t.threat, rep["kinds"], protect=False)
        protected_innocent = {r["unit"] for r in rep["rows"] if r.get("after_hide") and not r["y"] and r["ref"]}
        for u in units:
            k = rep["kinds"].get(u, "other")
            n_kinds[k] += 1
            if u in flags:
                (caught_kinds if u in rep["abusive"] else fa_kinds)[k] += 1
        totals["relationships"] += len(units)
        totals["payments"] += len(w.txs)
        totals["abusive"] += ev["abusive_relationships"]
        totals["caught"] += ev["caught"]
        totals["innocent"] += ev["innocent_relationships"]
        totals["false_alarms"] += ev["false_alarms"]
        totals["innocent_messages"] += msg["innocent_messages"]
        totals["innocent_hidden"] += round(msg["innocent_hidden_share"] * msg["innocent_messages"])
        totals["innocent_hidden_without_protect"] += round(msg_np["innocent_hidden_share"] * msg_np["innocent_messages"])
        totals["innocent_relationships_protect_hidden"] += len(protected_innocent)
        totals["abusive_messages"] += msg["abusive_messages"]
        totals["abusive_hidden"] += round(msg["abusive_hidden_share"] * msg["abusive_messages"])
        if args.chunks == 1:
            single = {**ev, "messages": msg, "messages_without_protect_after_hide": msg_np}
        print(f"chunk {c + 1}/{args.chunks}: {len(units)} relationships, {len(w.txs)} payments, "
              f"{ev['false_alarms']} false alarms, {ev['caught']}/{ev['abusive_relationships']} caught "
              f"({time.time() - t0:.0f}s)", flush=True)

    lo, hi = fpr_bounds(totals["false_alarms"], max(totals["innocent"], 1))
    out = {"tag": args.tag, "model": Path(args.model).name, "model_features": n_feat, "seed": args.seed,
           "chunks": args.chunks, "totals": totals,
           "false_alarm_rate": totals["false_alarms"] / max(totals["innocent"], 1),
           "false_alarm_rate_95ci": [lo, hi],
           "innocent_messages_hidden_share": totals["innocent_hidden"] / max(totals["innocent_messages"], 1),
           "false_alarms_by_kind": dict(fa_kinds.most_common()),
           "flag_rate_by_kind": {k: round((caught_kinds[k] + fa_kinds[k]) / n, 4) for k, n in sorted(n_kinds.items())},
           "seconds": round(time.time() - t0, 1)}
    if single:
        out["evaluation"] = single
    recall = args.recall if args.recall is not None else (totals["caught"] / totals["abusive"] if totals["abusive"] else None)
    if recall is not None:
        out["recall_used"] = recall
        out["precision_at_prevalence"] = {
            f"1 in {int(round(1 / p)):,}": {"measured": round(precision_at(recall, out["false_alarm_rate"], p), 4),
                                           "worst_case_95": round(precision_at(recall, hi, p), 4)}
            for p in (0.01, 0.001, 0.0001)}
    (ART / f"eval_{args.tag}.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({k: v for k, v in out.items() if k not in ("evaluation",)}, indent=1, default=str), flush=True)


if __name__ == "__main__":
    main()
