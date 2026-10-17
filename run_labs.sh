#!/usr/bin/env bash
# Start the two mandatory Datathon labs (Jupyter Lab with the 4 demo notebooks) on the LinuxONE VM.
#   ssh ibmz 'bash -s' < run_labs.sh     from your laptop
#   bash run_labs.sh                     or on the VM itself
# Same image as github.com/linuxone-community-cloud/jupyter-lab-ml, run with podman (already on the VM).
# Jupyter stays behind the VM firewall (only SSH is open); reach it through the SSH tunnel printed at the end.
set -euo pipefail

IMAGE=registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x:latest
NAME=notebook
PORT=38888

command -v podman >/dev/null || { sudo apt-get update -qq && sudo apt-get install -y -qq podman; }
mkdir -p ~/shared && chmod a+w ~/shared

echo "== 1/3 pull lab image (a few GB, first run only) =="
podman pull -q "$IMAGE"

start() {
  podman run -d --network host --name "$NAME" -v "$HOME/shared:/home/ibm-user/shared:z" "$@" "$IMAGE" >/dev/null
}
running() { podman ps --format '{{.Names}}' | grep -qx "$NAME"; }
# Any HTTP answer (even a redirect to the login page) means Jupyter is listening.
wait_up() {
  for _ in $(seq 90); do
    curl -s -o /dev/null -m 3 "http://127.0.0.1:$PORT/" && return 0
    running || return 1
    sleep 2
  done
  return 1
}

echo "== 2/3 start Jupyter Lab =="
if running && wait_up; then
  echo "already running"
else
  podman rm -f "$NAME" >/dev/null 2>&1 || true
  start
  if ! wait_up; then
    # The lab README notes some VMs need this flag.
    echo "first start failed, retrying with seccomp=unconfined"
    podman logs --tail 20 "$NAME" 2>&1 || true
    podman rm -f "$NAME" >/dev/null 2>&1 || true
    start --security-opt seccomp=unconfined
    wait_up || { podman logs --tail 40 "$NAME" 2>&1; echo "Jupyter did not start"; exit 1; }
  fi
fi

# Jupyter masks the token in its logs, so check the image's documented default instead.
TOKEN=Your_Token1
[ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/api/status?token=$TOKEN")" = 200 ] \
  || TOKEN="(not the default; see: podman exec $NAME jupyter server list)"
echo "== 3/3 ready =="
echo "On your laptop, open the tunnel and leave that window open:"
echo "  ssh -N -L $PORT:localhost:$PORT ibmz"
echo "Then browse to http://localhost:$PORT   token: $TOKEN"
