#!/usr/bin/env bash
# Start (or restart) the AgentGuard web app on the VM.
#   bash run.sh            private: 127.0.0.1:8080, reach it with an SSH tunnel
#   bash run.sh public     0.0.0.0:8080, needs port 8080 opened in iptables
set -euo pipefail
cd "$(dirname "$0")"
HOST=127.0.0.1
[ "${1:-}" = "public" ] && HOST=0.0.0.0
pkill -f '[u]vicorn agentguard' || true
setsid nohup ~/venv/bin/uvicorn agentguard.app.main:app --host "$HOST" --port 8080 \
  > server.log 2>&1 < /dev/null &
sleep 5
tail -n 3 server.log
echo "AgentGuard on http://$HOST:8080"
