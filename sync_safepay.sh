#!/usr/bin/env bash
# Push SafePay to the LinuxONE VM. Usage: bash sync_safepay.sh
set -euo pipefail
cd "$(dirname "$0")"
tar -cf - --exclude='__pycache__' --exclude='*.pyc' \
    safepay tests_safepay $(ls SAFEPAY*.md setup_safepay.sh run_safepay.sh Containerfile 2>/dev/null) \
  | ssh -o BatchMode=yes ibmz "mkdir -p ~/safepay && tar -xf - -C ~/safepay"
echo "synced -> ibmz:~/safepay"
