# SafePay: guide for judges

There are two ways to try SafePay:

- **Option A: the live demo** on the IBM z17 (LinuxONE Community Cloud). Nothing to install, and IBM Granite is running. About 5 minutes.
- **Option B: run it yourself** from this repository on any laptop (Windows, macOS or Linux, x86 or ARM), or on a LinuxONE VM. About 30 minutes, most of it training.

The demo holds only fictional people and data.

---

## Option A: the live demo

1. Open **http://148.100.99.226:8090** in a desktop browser (Chrome, Edge, Firefox).
2. Sign in with user name **`judge`** and the password given in the submission description.
3. If the three columns stack on top of each other, zoom out once (**Ctrl + −** / **Cmd + −**).

Then follow the walkthrough below.

---

## Walkthrough (5 minutes)

The home page has three columns: the **sender's app** (left), what **SafePay** decided (middle) and the **recipient's app** (right). Every payment goes through the real screening service as an ISO 20022 `pacs.008` message.

Press **Reset** first if someone has used the demo before you. Click each scenario **once**: a second click adds more payments to the same case.

| # | Do this | What to look for |
|---|---|---|
| 1 | Click **Threat split across 9 payments** | No single reference is a threat on its own. The middle column shows the joined text ("I am going to find you and kill you"), and all nine messages are hidden in Sam's app. The money is still delivered. |
| 2 | Click **Polite but menacing (coercive control)** | "Saw you at the gym today" scores near zero as a message, but the relationship model (penny payments, at night, never answered) opens a case. |
| 3 | Click **Friends' banter (should pass)** | Rude-sounding jokes between friends are delivered as normal: no case. |
| 4 | Type your own message in the sender's app and press **Send payment** | Try evasion tricks: `k.i.l.l`, `d3ad`, `k i l l`, or a 🔪 emoji. The middle column shows the cleaned-up text and the words that caused the flag. |
| 5 | In the recipient's app, open **Message settings** | The victim's side: safe-contact rules (choose "In the app only", tick "Someone else may see my phone", **Save**), support numbers, and **Allow a record** for a sender, e.g. for a protection order. Try **Quick exit** (or Esc). |
| 6 | Open **/analyst** (link at the top of the page) | The specialist team's ranked queue. Pick a case: the customer's safe-contact rules are shown first, then the payment timeline, scores and reasons. **Ask Granite** queues an IBM Granite 3.3 brief, written on the same machine (about a minute). |
| 7 | On a case where the customer allowed a record (step 5), press **Prepare signed evidence pack** | The pack opens at `/pack/{id}`. Press **Verify** (signature and fingerprint check out), then **Simulate tampering** (one character changed in a copy, and verification fails). |
| 8 | Open **/system** | Every number in the README, read from the training, evaluation and benchmark outputs, plus the live model version. |

Other scenarios to try: **Child support with abuse** (money that must arrive, with abusive references) and **Hidden, then switches account** (after Sam hides a sender, messages from brand-new senders are hidden by default for 14 days).

---

## Option B: run it yourself

You need **Python 3.10 or newer** and **git**. IBM Granite is optional: without it everything works except the Granite brief and second opinion in the analyst console. Off IBM Z, the relationship model trains with scikit-learn instead of IBM Snap ML (same features, same policy).

Commands are for macOS / Linux. On Windows (PowerShell), use `py` for the first `python`, and `.venv\Scripts\python` instead of `.venv/bin/python`.

### 1. Get the code and install

```bash
git clone https://github.com/suhasadiga06/IBMZ.git
cd IBMZ
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt pyarrow huggingface_hub
```

### 2. Download the data

The trained models are not in the repository, so you build them from public data: the [Civil Comments](https://huggingface.co/datasets/google/civil_comments) dataset (CC0) and Jigsaw's bias-test sentences (Apache-2.0). About 420 MB to download.

```bash
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('google/civil_comments', repo_type='dataset', local_dir='civil_comments_parquet', allow_patterns='*.parquet')"
.venv/bin/python -m safepay.scripts.convert_parquet civil_comments_parquet/data data
curl -sSL -o data/bias_templates_en.csv https://raw.githubusercontent.com/conversationai/unintended-ml-bias-analysis/main/sentence_templates/en_sentence_templates.csv
```

You should now have `data/cc_train.csv`, `data/cc_val.csv`, `data/cc_test.csv` and `data/bias_templates_en.csv`.

### 3. Train the two models

```bash
.venv/bin/python -m safepay.scripts.train_text
.venv/bin/python -m safepay.scripts.train_rel
```

On a typical laptop this takes about 15–20 minutes in total: the message model on 260,000 comments (about 8 minutes), then three synthetic payment worlds replayed through it to train the relationship model. Both write to `artifacts/`.

### 4. Start SafePay

```bash
.venv/bin/python -m uvicorn safepay.app.main:app --port 8090
```

Open **http://localhost:8090** and follow the [walkthrough](#walkthrough-5-minutes). No password is needed locally. Encryption and signing keys are created on first start in `~/.safepay/`, and the case database is `safepay.db` in the repository folder (delete it to start over).

The **/system** page shows the numbers from your own training run. To also see the z17 benchmark and the outside check against Commonwealth Bank's model, copy those results from the VM run into `artifacts/`:

```bash
cp results/benchmark.json results/cba_validation.json results/eval_*.json results/granite_eval_*.json artifacts/
```

### 5. Optional: IBM Granite

Run any llama.cpp server with [IBM Granite 3.3 2B Instruct (GGUF)](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct-GGUF) on port 8081, for example:

```bash
llama-server -m granite-3.3-2b-instruct-Q4_K_M.gguf --host 127.0.0.1 --port 8081 -c 4096
```

SafePay looks for it at `http://127.0.0.1:8081` (change with `SAFEPAY_LLM_URL`).

### 6. Tests

```bash
.venv/bin/python -m pytest -q tests_safepay
```

36 tests: the hash chain catching edits and deletions, evidence-pack signatures, consent gating, the security controls and encryption at rest. The model tests are skipped until step 3 has been run.

### Try the bank-facing API

With the service running, screen a real ISO 20022 message yourself. Interactive API docs are at **http://localhost:8090/docs**: open `POST /v1/screen`, press *Try it out*, and paste a `pacs.008` document.

---

## On IBM Z / LinuxONE

The full build used for the results (native s390x packages, IBM Snap ML on the Telum II AI accelerator, Granite converted to big-endian and built with llama.cpp for z16/z17) is two scripts on an Ubuntu 24.04 LinuxONE Community Cloud VM. See [Run it](README.md#run-it) in the README.

```bash
bash setup_safepay.sh
bash run_safepay.sh
```

## If something goes wrong

| Problem | Fix |
|---|---|
| The live demo doesn't load | The VM may have been restarted. Use Option B, or contact the team through the Hack2Skill submission. |
| `FileNotFoundError: artifacts/text.joblib` | Run step 3 (training) first. |
| `FileNotFoundError: data/cc_val.csv` | Run step 2. Check the Parquet files are under `civil_comments_parquet/data/`. |
| "Granite couldn't write this brief" in the analyst console | Expected without step 5. Everything else works. |
| `429 Too Many Requests` | The public demo is rate-limited. Wait a minute. |
| The page looks cramped | Zoom out once, or widen the window. The demo is designed for a desktop screen. |
