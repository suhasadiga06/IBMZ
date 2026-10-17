"""Convert Civil Comments Parquet files to the short-text CSVs SafePay uses.

Run on any machine that has pyarrow (it has no s390x build, so not on the
LinuxONE VM), then copy the CSVs to data/ on the VM:

  python -m safepay.scripts.convert_parquet <dir with the 4 .parquet files> <out dir>

Keeps texts of at most MAX_CHARS characters (a payment reference is at most
140 in ISO 20022; we keep a little longer for training variety), removes
duplicates, and preserves the official train / validation / test split.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

MAX_CHARS = 300
LABELS = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit"]


def convert(src: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    for split, pattern in (("test", "test-*.parquet"), ("val", "validation-*.parquet"), ("train", "train-*.parquet")):
        frames = []
        for f in sorted(src.glob(pattern)):
            t = pq.read_table(f, columns=["text", *LABELS])
            t = t.filter(pc.less_equal(pc.utf8_length(t["text"]), MAX_CHARS))
            frames.append(t.to_pandas())
        import pandas as pd
        df = pd.concat(frames, ignore_index=True)
        df["text"] = df["text"].str.strip()
        df = df[df.text.astype(bool)].drop_duplicates("text")
        # A text seen in an earlier split (test first) is dropped from later ones.
        df = df[~df.text.isin(seen)]
        seen.update(df.text)
        df[LABELS] = df[LABELS].round(4)
        path = out / f"cc_{split}.csv"
        df.to_csv(path, index=False)
        print(f"{path.name}: {len(df)} rows | toxic {int((df.toxicity >= 0.5).sum())} | "
              f"threat>=0.5 {int((df.threat >= 0.5).sum())} | threat>=0.3 {int((df.threat >= 0.3).sum())} | "
              f"<=140 chars {int((df.text.str.len() <= 140).sum())}")


if __name__ == "__main__":
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
