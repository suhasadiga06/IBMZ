#!/usr/bin/env bash
# Start (or restart) ChargeGuard on the VM.
#   bash run_chargeguard.sh          service on 127.0.0.1:8095 (reach it with an SSH tunnel)
#   bash run_chargeguard.sh demo     also replay the held-out stations (demo fault mix 25%)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p ~/logs

if ! curl -s -m 2 http://127.0.0.1:8081/health > /dev/null; then
  echo "Granite is not answering on 127.0.0.1:8081: incident notes will use templates."
  echo "(Start it with: bash ~/safepay/run_safepay.sh)"
fi

pkill -f '[c]hargeguard.sim.replay' || true
pkill -f '[u]vicorn chargeguard' || true
# An open dashboard holds a live event stream; don't let it keep the old service alive.
for _ in $(seq 5); do pgrep -f '[u]vicorn chargeguard' > /dev/null || break; sleep 1; done
pkill -9 -f '[u]vicorn chargeguard' || true
# One reading at a time, Snap ML's CPU tree engine was fastest on the z17 (results/benchmark.json).
export CHARGEGUARD_TREE_FORMAT="${CHARGEGUARD_TREE_FORMAT:-compress_trees}"
setsid nohup ~/venv/bin/uvicorn chargeguard.app.main:app --host 127.0.0.1 --port 8095 --timeout-graceful-shutdown 3 \
  > ~/logs/chargeguard.log 2>&1 < /dev/null &
for _ in $(seq 30); do curl -s -m 2 http://127.0.0.1:8095/healthz > /dev/null && break; sleep 1; done
curl -s http://127.0.0.1:8095/healthz && echo

if [ "${1:-}" = "demo" ]; then
  setsid nohup ~/venv/bin/python -m chargeguard.sim.replay --minute-seconds 0.5 --fault-share 0.25 \
    > ~/logs/chargeguard_replay.log 2>&1 < /dev/null &
  echo "replay of the held-out stations started (log: ~/logs/chargeguard_replay.log)"
fi
echo "ChargeGuard on http://127.0.0.1:8095  (live /, evidence /evidence)"
echo "From your laptop: ssh -N -L 8095:localhost:8095 ibmz   then open http://localhost:8095"
