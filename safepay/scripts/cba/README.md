# Checking our simulator with Commonwealth Bank's production model

CBA's open-source model ([h2oai/aitd](https://github.com/h2oai/aitd), Apache-2.0) was trained on
real, expert-labelled abuse cases. We score our synthetic test world with it: if it ranks our
synthetic abusers above our innocent relationships, the abuse our simulator produces looks like
real abuse to a model that has seen real abuse.

Two steps, because PyTorch has no s390x build but Java does:

1. **Features, on any x86/ARM machine** (needs ~1.5 GB RAM, a few minutes on a laptop CPU):
   ```bash
   # folder layout next to run_cba_features.py:
   #   aitd/src/*.py   CBA's src/ folder (from their repo), aitd/logs/
   #   models/toxic_bias-4e693588.ckpt   (Detoxify v0.1-alpha release, the checkpoint CBA uses)
   #   cba_input.csv   from: python3 -m safepay.scripts.export_cba   (on the VM)
   python -m venv env
   env/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
   env/bin/pip install "transformers>=4.40,<4.46" detoxify vaderSentiment wordninja pandas python-dateutil
   env/bin/python run_cba_features.py          # writes features_YYYYMM_lagN.csv
   ```
2. **Scoring, on the LinuxONE VM** with plain Java (`openjdk-17-jre-headless`) and CBA's MOJO files
   (`h2o-genmodel.jar`, `CBA_AITD_Short.zip`, `CBA_AITD_Long.zip` from their `models/` folder):
   ```bash
   python3 -m safepay.scripts.score_cba --cba-dir ~/cba --features-dir ~/cba/features
   ```

Changes from CBA's pipeline: the neural models score each unique reference once instead of
every payment (outputs cached in `nlp_cache.csv`); no multiprocessing; H2O/GPUtil/pandarallel
are stubbed (scoring runs in Java); NumPy 2's `array_split` change is worked around; and the
checkpoint is loaded with the official `detoxify` package instead of CBA's copy of it. One edit
to CBA's own code is needed for pandas 2, in `src/process_transactions.py` line 107:

```python
cols_to_duplicate = list(set(reference).intersection(final_edges.columns))   # was: set(...)
```

pandas 2 no longer accepts a set as a column indexer. Columns are merged by name, so the order
doesn't matter. Run with `SMOKE=1` first to test the aggregation path in seconds with placeholder
scores. Otherwise the feature code is CBA's, unchanged.
