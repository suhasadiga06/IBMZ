"""Export a synthetic test world in Commonwealth Bank's model input format.

Used to score our simulator with CBA's open-source production model
(github.com/h2oai/aitd), which was trained on real, expert-labelled abuse.
If that model separates our synthetic abusers from innocent relationships the
way it separates real ones, our simulator's abuse patterns are realistic.

Writes data/cba_input.csv (tx_description, sender_id, receiver_id, tx_date,
amount) and data/cba_labels.csv (sender_id, receiver_id, label, kind).
Usage: python3 -m safepay.scripts.export_cba
"""
from __future__ import annotations

import csv
from pathlib import Path

from safepay.scripts.train_rel import SIZES
from safepay.sim import world as W

DATA = Path(__file__).resolve().parents[2] / "data"


def main() -> None:
    n_ok, n_bad, seed = SIZES["test"]
    w = W.build("test", n_ok, n_bad, seed)
    ids = {acct: f"U{i:05d}" for i, acct in enumerate(sorted(w.customers))}
    with (DATA / "cba_input.csv").open("w", newline="", encoding="utf-8") as f:
        out = csv.writer(f)
        out.writerow(["tx_description", "sender_id", "receiver_id", "tx_date", "amount"])
        for t in w.txs:
            out.writerow([t.reference, ids[t.sender], ids[t.recipient], t.ts.strftime("%Y-%m-%d %H:%M:%S"), f"{t.amount:.2f}"])
    with (DATA / "cba_labels.csv").open("w", newline="", encoding="utf-8") as f:
        out = csv.writer(f)
        out.writerow(["sender_id", "receiver_id", "label", "kind"])
        for r in w.relationships:
            out.writerow([ids[r.sender], ids[r.recipient], r.label, r.kind])
            out.writerow([ids[r.recipient], ids[r.sender], 0, r.kind + ("_reverse" if r.label else "")])
    print(f"seed {seed}: {len(w.txs)} payments, {len(w.relationships)} relationships, "
          f"{sum(r.label for r in w.relationships)} abusive, {len({t.reference for t in w.txs})} unique references")


if __name__ == "__main__":
    main()
