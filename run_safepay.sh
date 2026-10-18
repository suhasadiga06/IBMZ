#!/usr/bin/env bash
# Start (or restart) Granite and SafePay on the VM.
#   bash run_safepay.sh              demo on 127.0.0.1:8090 (reach it with an SSH tunnel)
#   bash run_safepay.sh production   no demo pages; API keys and analyst tokens required
#   bash run_safepay.sh container    the Podman image instead (demo, local only)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p ~/logs
MODE="${1:-local}"

if ! curl -s -m 2 http://127.0.0.1:8081/health > /dev/null; then
  # Granite listens on localhost only, in every mode: it is never reachable from outside.
  setsid nohup ~/llama.cpp/build/bin/llama-server -m ~/models/granite-3.3-2b-instruct-Q4_K_M-be.gguf \
    --host 127.0.0.1 --port 8081 -t "$(nproc)" -c 4096 --no-webui > ~/logs/granite_server.log 2>&1 < /dev/null &
  echo "Granite starting on 127.0.0.1:8081"
fi

case "$MODE" in
  production)
    ~/venv/bin/python -m safepay.scripts.make_keys > /dev/null
    export SAFEPAY_MODE=production ;;
  local|container) ;;
  *) echo "unknown mode: $MODE"; exit 1 ;;
esac

pkill -f '[s]afepay.app.main' || true
podman rm -f safepay > /dev/null 2>&1 || true
if [ "$MODE" = "container" ]; then
  podman run -d --name safepay --network host -v safepay-data:/var/lib/safepay localhost/safepay
else
  setsid nohup ~/venv/bin/uvicorn safepay.app.main:app --host 127.0.0.1 --port 8090 --no-server-header \
    > ~/logs/safepay_server.log 2>&1 < /dev/null &
fi
sleep 8
curl -s http://127.0.0.1:8090/healthz && echo
echo "SafePay ($MODE) on http://127.0.0.1:8090"
exit 0
