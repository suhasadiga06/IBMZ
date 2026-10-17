#!/usr/bin/env bash
# Build ChargeGuard from scratch on a LinuxONE Community Cloud VM (Ubuntu 24.04, s390x).
#
#   bash setup_chargeguard.sh          everything below
#   bash run_chargeguard.sh [demo]     start the service (and the replay of held-out stations)
#
# Incident notes use IBM Granite through llama.cpp on 127.0.0.1:8081 (the same server
# SafePay's setup builds); without it, ChargeGuard falls back to template notes.
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1/6 system packages (native s390x builds) =="
# numpy/scipy/scikit-learn have no s390x wheels on PyPI: use Ubuntu's native builds.
# libzdnn0 is IBM's library for the on-chip AI accelerator, used by Snap ML.
sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  python3-venv python3-pip python3-numpy python3-scipy python3-sklearn python3-pandas python3-joblib libzdnn0

echo "== 2/6 Python environment =="
[ -x ~/venv/bin/python ] || python3 -m venv --system-site-packages ~/venv
~/venv/bin/pip install -q fastapi uvicorn websockets httpx pytest snapml openpyxl

echo "== 3/6 data =="
mkdir -p data artifacts results
if [ ! -f data/ev30.csv.gz ] && [ -f ~/evdata/ev30.csv.gz ]; then cp ~/evdata/ev30.csv.gz data/; fi
~/venv/bin/python -m chargeguard.scripts.prepare_data
~/venv/bin/python -m chargeguard.scripts.fetch_tariff

echo "== 4/6 train and evaluate (Snap ML on IBM Z) =="
nice ~/venv/bin/python -m chargeguard.scripts.train

echo "== 5/6 tests =="
~/venv/bin/python -m pytest -q tests

echo "== 6/6 benchmark =="
~/venv/bin/python -m chargeguard.scripts.benchmark
echo "Done. Start with: bash run_chargeguard.sh demo"
