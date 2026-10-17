#!/usr/bin/env bash
# One-shot setup on a fresh LinuxONE Community Cloud VM (Ubuntu 24.04, s390x).
#   bash setup_vm.sh            install, fetch data, train, test
#   bash run.sh                 start the web app
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1/5 system packages =="
# numpy, scipy and scikit-learn have no s390x wheels on PyPI, so take Ubuntu's native builds.
sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  python3-venv python3-pip python3-numpy python3-scipy python3-sklearn python3-pandas python3-joblib

echo "== 2/5 virtualenv =="
# --system-site-packages lets the venv see the apt-installed ML stack.
python3 -m venv --system-site-packages ~/venv
~/venv/bin/pip install -q fastapi uvicorn httpx pytest

echo "== 3/5 datasets =="
~/venv/bin/python -m agentguard.scripts.fetch_data --rows 60000

echo "== 4/5 train =="
~/venv/bin/python -m agentguard.scripts.train

echo "== 5/5 test =="
~/venv/bin/python -m pytest -q tests
~/venv/bin/python -m agentguard.scripts.demo_cli
echo "Done. Start the app with: bash run.sh"
