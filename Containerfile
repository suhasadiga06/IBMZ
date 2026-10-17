# SafePay screening service, built natively on IBM Z (s390x).
#   podman build -t safepay -f Containerfile .
#   podman run -d --name safepay --network host -v safepay-data:/var/lib/safepay safepay
# Granite runs as a separate llama.cpp server; point SAFEPAY_LLM_URL at it.
FROM docker.io/library/ubuntu:24.04

# numpy/scipy/scikit-learn have no s390x wheels on PyPI: use Ubuntu's native builds.
# libzdnn0 is IBM's library for the on-chip AI accelerator, used by Snap ML.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip python3-numpy python3-scipy python3-sklearn python3-pandas \
      python3-joblib python3-cryptography libzdnn0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv --system-site-packages /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir fastapi uvicorn snapml

WORKDIR /app
COPY safepay ./safepay
COPY artifacts/text.joblib artifacts/rel.joblib artifacts/metrics_*.json artifacts/benchmark.json ./artifacts/
COPY data/cc_test.csv ./data/

ENV SAFEPAY_DB=/var/lib/safepay/safepay.db \
    SAFEPAY_KEY_FILE=/var/lib/safepay/key \
    SAFEPAY_SIGN_KEY_FILE=/var/lib/safepay/sign_key \
    SAFEPAY_LLM_URL=http://127.0.0.1:8081 \
    PYTHONUNBUFFERED=1
VOLUME /var/lib/safepay
EXPOSE 8090
HEALTHCHECK --interval=30s --timeout=5s CMD python3 -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8090/healthz',timeout=4)"
USER 1000
CMD ["/opt/venv/bin/uvicorn", "safepay.app.main:app", "--host", "0.0.0.0", "--port", "8090"]
