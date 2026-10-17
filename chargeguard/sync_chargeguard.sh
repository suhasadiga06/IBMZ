#!/usr/bin/env bash
# Push ChargeGuard to the LinuxONE VM, or fetch the VM's results back.
#   bash sync_chargeguard.sh           laptop -> ibmz:~/chargeguard (code only)
#   bash sync_chargeguard.sh results   ibmz:~/chargeguard/results -> ./results
set -euo pipefail
cd "$(dirname "$0")"
if [ "${1:-}" = "results" ]; then
  mkdir -p results
  scp -q -o BatchMode=yes "ibmz:~/chargeguard/results/*.json" results/
  echo "fetched ibmz:~/chargeguard/results -> results/"
  exit 0
fi
tar -cf - --exclude='__pycache__' --exclude='*.pyc' \
    chargeguard tests setup_chargeguard.sh run_chargeguard.sh requirements.txt README.md 2>/dev/null \
  | ssh -o BatchMode=yes ibmz "mkdir -p ~/chargeguard && tar -xf - -C ~/chargeguard"
echo "synced -> ibmz:~/chargeguard"
