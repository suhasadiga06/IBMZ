#!/usr/bin/env bash
# Start (or restart) Granite and SafePay on the VM.
#   bash run_safepay.sh             SafePay on 127.0.0.1:8090 (reach it with an SSH tunnel)
#   bash run_safepay.sh container   SafePay from the Podman image instead
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p ~/logs

if ! curl -s -m 2 http://127.0.0.1:8081/health > /dev/null; then
  setsid nohup ~/llama.cpp/build/bin/llama-server -m ~/models/granite-3.3-2b-instruct-Q4_K_M-be.gguf \
    --host 127.0.0.1 --port 8081 -t "$(nproc)" -c 4096 --no-webui > ~/logs/granite_server.log 2>&1 < /dev/null &
  echo "Granite starting on 127.0.0.1:8081"
fi

pkill -f '[s]afepay.app.main' || true
podman rm -f safepay > /dev/null 2>&1 || true
if [ "${1:-}" = "container" ]; then
  podman run -d --name safepay --network host -v safepay-data:/var/lib/safepay localhost/safepay
else
  setsid nohup ~/venv/bin/uvicorn safepay.app.main:app --host 127.0.0.1 --port 8090 \
    > ~/logs/safepay_server.log 2>&1 < /dev/null &
fi
sleep 8
curl -s http://127.0.0.1:8090/healthz && echo
echo "SafePay on http://127.0.0.1:8090  (demo /, analyst /analyst, evidence /system)"
