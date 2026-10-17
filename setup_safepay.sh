#!/usr/bin/env bash
# Rebuild SafePay from scratch on a LinuxONE Community Cloud VM (Ubuntu 24.04, s390x).
#
#   bash setup_safepay.sh           everything below
#   bash run_safepay.sh             start Granite + SafePay
#
# Civil Comments: pyarrow has no s390x build, so the Parquet -> CSV step runs on
# any other machine (see safepay/scripts/convert_parquet.py). Put the three CSVs
# (cc_train.csv, cc_val.csv, cc_test.csv) in data/ before running this.
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1/6 system packages (native s390x builds) =="
sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  python3-venv python3-pip python3-numpy python3-scipy python3-sklearn python3-pandas python3-joblib \
  python3-cryptography libzdnn0 build-essential cmake libopenblas-dev podman

echo "== 2/6 Python environment =="
python3 -m venv --system-site-packages ~/venv
~/venv/bin/pip install -q fastapi uvicorn httpx pytest snapml tqdm pyyaml

echo "== 3/6 bias test set =="
[ -f data/bias_templates_en.csv ] || curl -sSL -o data/bias_templates_en.csv \
  https://raw.githubusercontent.com/conversationai/unintended-ml-bias-analysis/main/sentence_templates/en_sentence_templates.csv
for f in cc_train.csv cc_val.csv cc_test.csv; do
  [ -f "data/$f" ] || { echo "missing data/$f - see the note at the top of this script"; exit 1; }
done

echo "== 4/6 train and evaluate =="
~/venv/bin/python -m safepay.scripts.train_text
~/venv/bin/python -m safepay.scripts.train_rel

echo "== 5/6 IBM Granite via llama.cpp (z16 target: GCC 13 cannot target z17 yet) =="
if [ ! -x ~/llama.cpp/build/bin/llama-server ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git ~/llama.cpp
  cmake -S ~/llama.cpp -B ~/llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS \
    -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF -DCMAKE_C_FLAGS="-march=z16 -mtune=z16" -DCMAKE_CXX_FLAGS="-march=z16 -mtune=z16"
  cmake --build ~/llama.cpp/build --config Release -j "$(nproc)" --target llama-server llama-bench
fi
M=~/models/granite-3.3-2b-instruct-Q4_K_M
if [ ! -f "$M-be.gguf" ]; then
  mkdir -p ~/models
  curl -sSL -o "$M.gguf" https://huggingface.co/ibm-granite/granite-3.3-2b-instruct-GGUF/resolve/main/granite-3.3-2b-instruct-Q4_K_M.gguf
  cp "$M.gguf" "$M-be.gguf"
  # The upstream converter needs numpy >= 2.1 for reshape(copy=False); Ubuntu has 1.26,
  # where a plain reshape of the memory-mapped tensor is the same in-place view.
  sed 's/, copy=False)/)/g' ~/llama.cpp/gguf-py/gguf/scripts/gguf_convert_endian.py > /tmp/convert_endian.py
  yes YES | PYTHONPATH=~/llama.cpp/gguf-py ~/venv/bin/python /tmp/convert_endian.py "$M-be.gguf" big
fi

echo "== 6/6 tests and benchmark =="
~/venv/bin/python -m pytest -q tests_safepay
echo "Done. Start with: bash run_safepay.sh   then: ~/venv/bin/python -m safepay.scripts.benchmark --url http://127.0.0.1:8090 --granite"
