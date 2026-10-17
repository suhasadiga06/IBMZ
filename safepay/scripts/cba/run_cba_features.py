"""Compute Commonwealth Bank's model features for our test world, on a laptop.

Runs CBA's own feature code (github.com/h2oai/aitd, src/) unchanged, with
three practical changes:
  * the expensive neural models run once per unique reference, not per payment
  * no multiprocessing (CBA starts 4 processes; this laptop is small)
  * h2o / GPUtil / pandarallel are stubbed: scoring happens on the VM in Java
Usage: env/Scripts/python run_cba_features.py
"""
import os
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["HF_HOME"] = os.path.join(HERE, "hf")
os.environ["TORCH_HOME"] = os.path.join(HERE, "torch")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch  # noqa: E402

torch.set_num_threads(2)            # keep the laptop usable
_load = torch.load
torch.load = lambda *a, **k: _load(*a, **{"weights_only": False, **k})   # official Detoxify release file

sys.modules["GPUtil"] = types.ModuleType("GPUtil")
sys.modules["h2o"] = types.ModuleType("h2o")
_pp = types.ModuleType("pandarallel")
_pp.pandarallel = types.SimpleNamespace(initialize=lambda **k: None)
sys.modules["pandarallel"] = _pp

sys.path.insert(0, os.path.join(HERE, "aitd"))
os.chdir(os.path.join(HERE, "aitd"))          # CBA's config logs to logs/aitd.log

import detoxify as official  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from src import config, publicmodels, simpletext, sentiment  # noqa: E402
from src import process_transactions as pt  # noqa: E402

CKPT = os.path.join(HERE, "models", "toxic_bias-4e693588.ckpt")


class SameCheckpoint:
    """CBA's checkpoint (Detoxify v0.1-alpha 'toxic_bias'), loaded by the official
    Detoxify package instead of CBA's copy, which expects a local roberta-base clone."""

    def __init__(self, burt_model_loc=None, model_type="unbiased", checkpoint=None, device="cpu"):
        self._d = official.Detoxify(checkpoint=checkpoint or CKPT, device=device)
        self.class_names = self._d.class_names

    def predict(self, text):
        return self._d.predict(text)


publicmodels.detoxify.Detoxify = SameCheckpoint
config.tox_model_path = CKPT


class _NumpyForCBA:
    """CBA batches with np.array_split(DataFrame, n). Under NumPy 2 that returns
    plain arrays, which CBA's code then fails on (and silently swallows). Split
    DataFrames by row ranges, as older NumPy did; everything else is NumPy."""

    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def array_split(obj, n):
        if isinstance(obj, pd.DataFrame):
            bounds = np.linspace(0, len(obj), int(n) + 1).astype(int)
            return [obj.iloc[a:b] for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
        return np.array_split(obj, n)


publicmodels.np = _NumpyForCBA()
Z = config.txn_data_ind_col

TOX_COLS = ["toxicity", "severe_toxicity", "obscene", "identity_attack", "insult", "threat", "sexual_explicit"]

# Windows to score: month, lag (CBA_Short uses lag 0, CBA_Long uses lag 2).
RUNS = [(2026, 5, 0), (2026, 6, 0), (2026, 7, 0), (2026, 8, 0), (2026, 8, 2)]


def main() -> None:
    t0 = time.time()
    txn = pd.read_csv(os.path.join(HERE, "cba_input.csv"))
    txn["tx_description"] = txn["tx_description"].fillna("").astype(str)
    txn = pt.pre_process_txn(txn)

    uniq = pd.DataFrame({"tx_description": sorted(txn.tx_description.unique())})
    uniq["sender_id"], uniq["receiver_id"], uniq["amount"] = 0, 0, 0.0
    uniq["tx_date"] = txn.tx_date.iloc[0]
    uniq[Z] = uniq.index
    print(f"{len(txn)} payments, {len(uniq)} unique references", flush=True)

    cache = os.path.join(HERE, "nlp_cache.csv")
    if os.environ.get("SMOKE"):
        # Placeholder scores to test the aggregation path in seconds. Writes nothing.
        rng = np.random.default_rng(0)
        by_text = uniq[["tx_description"]].copy()
        for c in TOX_COLS:
            by_text[c] = rng.random(len(by_text))
        emo_cols = ["anger", "fear", "joy", "love", "sadness", "surprise", "neutral"]
        for c in emo_cols:
            by_text[c] = 0
        by_text["joy"] = 1
        tox_cols = TOX_COLS
    elif os.path.exists(cache):
        # The neural models are the slow part; their per-text outputs are reused across runs.
        by_text = pd.read_csv(cache, keep_default_na=False)
        tox_cols, emo_cols = TOX_COLS, [c for c in by_text.columns if c not in TOX_COLS + ["tx_description"]]
        print(f"loaded cached model outputs for {len(by_text)} texts", flush=True)
    else:
        tox = publicmodels.public_models(uniq).detoxify_model()
        # CBA's code logs model errors and returns empty frames; never continue on one.
        assert len(tox) == len(uniq), f"toxicity model returned {len(tox)} of {len(uniq)} rows: see aitd/logs/aitd.log"
        print(f"toxicity done {time.time() - t0:.0f}s, classes {list(tox.columns)[1:]}", flush=True)
        emo = publicmodels.get_emotion_model_results(uniq)
        assert Z in emo.columns and len(emo) == len(uniq), "emotion model failed: see aitd/logs/aitd.log"
        print(f"emotion done {time.time() - t0:.0f}s", flush=True)
        by_text = uniq[[Z, "tx_description"]].merge(tox, on=Z).merge(emo, on=Z).drop(columns=Z)
        tox_cols = [c for c in tox.columns if c != Z]
        emo_cols = [c for c in emo.columns if c != Z]
        assert tox_cols == TOX_COLS
        by_text.to_csv(cache, index=False)
    full = txn[[Z, "tx_description"]].merge(by_text, on="tx_description", how="left").fillna(0)
    tox_full, emo_full = full[[Z, *tox_cols]], full[[Z, *emo_cols]]

    _, text_df = simpletext.get_simple_text_features(txn.copy())
    sent_df = sentiment.get_sentiment_scores(txn.copy())
    prefinal = pt.merge_tox_amo_scores(tox_full, emo_full, text_df, sent_df)
    print(f"per-payment features done {time.time() - t0:.0f}s", flush=True)

    ym = txn.tx_date.dt.year * 100 + txn.tx_date.dt.month
    for year, month, lag in RUNS:
        # As CBA intends: give the pipeline only the scored month and its lag months.
        months = pt.get_prev_year_months(year, month, n=lag) + [year * 100 + month]
        keep = ym.isin(months)
        t_win = txn[keep].copy()
        p_win = prefinal[prefinal[Z].isin(t_win[Z])].copy()
        _, edge_df, final_set = sentiment.get_edge_features(t_win.copy(), p_win)
        _, final_edges = sentiment.get_time_features(edge_df, final_set)
        feats = pt.gen_features(final_edges, t_win.copy(), lag=lag, score_month=month, score_year=year)
        out = os.path.join(HERE, ("smoke_" if os.environ.get("SMOKE") else "") + f"features_{year}{month:02d}_lag{lag}.csv")
        feats.fillna(0).to_csv(out, index=False)
        print(f"wrote {os.path.basename(out)}: {len(feats)} relationships, {feats.shape[1]} columns", flush=True)
    print(f"all done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
