#!/usr/bin/env bash
# Push the project to the LinuxONE VM. Usage: bash sync.sh
set -euo pipefail
cd "$(dirname "$0")"
tar -cf - --exclude='__pycache__' --exclude='*.pyc' --exclude='.venv' \
    agentguard tests requirements.txt setup_vm.sh run.sh README.md 2>/dev/null \
  | ssh -o BatchMode=yes ibmz "mkdir -p ~/agentguard && tar -xf - -C ~/agentguard"
echo "synced -> ibmz:~/agentguard"
