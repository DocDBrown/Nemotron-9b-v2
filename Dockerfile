# syntax=docker/dockerfile:1
FROM dustynv/l4t-pytorch:r36.4.0

ARG DEBIAN_FRONTEND=noninteractive
ARG HF_TOKEN=""

# Cache locations inside the image
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=180 \
    PIP_RETRIES=10 \
    PIP_INDEX_URL=https://pypi.org/simple \
    HF_HOME=/opt/hf \
    HF_HUB_CACHE=/opt/hf/hub \
    TORCH_HOME=/opt/torch \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

ARG JETSON_WHEELS=https://pypi.jetson-ai-lab.io/jp6/cu126

WORKDIR /app

# --------------------------------------------------------
# Install OS build tools + Python deps
# --------------------------------------------------------
RUN << 'BASH'
set -eux

apt-get update
apt-get install -y --no-install-recommends \
    git build-essential cmake ninja-build pkg-config python3-dev \
    libopenblas-dev libffi-dev ca-certificates

rm -rf /var/lib/apt/lists/*

python3 -m pip install -U pip setuptools wheel

# --------------------------------------------------------
# Model and runtime deps
# --------------------------------------------------------
python3 -m pip install \
    "transformers==4.48.3" \
    "accelerate>=0.30.0" \
    safetensors sentencepiece huggingface_hub

# --------------------------------------------------------
# Web server deps
# --------------------------------------------------------
python3 -m pip install fastapi uvicorn

# --------------------------------------------------------
# Quantization (best effort)
# --------------------------------------------------------
python3 -m pip install --extra-index-url "${JETSON_WHEELS}" bitsandbytes || \
    python3 -m pip install bitsandbytes || true

# --------------------------------------------------------
# Install causal-conv1d first
# --------------------------------------------------------
python3 -m pip install causal-conv1d>=1.4.0 --no-build-isolation

# --------------------------------------------------------
# Install mamba-ssm with causal-conv1d extras and force build
# - disable isolation to ensure it sees the system CUDA + PyTorch
# --------------------------------------------------------
python3 -m pip install mamba-ssm[causal-conv1d] --no-build-isolation

# Optional: triton (not fatal, but sometimes needed for kernels)
python3 -m pip install --extra-index-url "${JETSON_WHEELS}" triton || \
    python3 -m pip install triton || true

# --------------------------------------------------------
# Pre-download model into HF cache
# --------------------------------------------------------
HF_TOKEN="${HF_TOKEN}" HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_HUB_DISABLE_XET=1 python3 - << 'PY'
import os, time
from huggingface_hub import snapshot_download

model_id="nvidia/NVIDIA-Nemotron-Nano-9B-v2"
token=os.environ.get("HF_TOKEN") or None
last_err=None

for attempt in range(1,8):
    try:
        snapshot_download(
            repo_id=model_id,
            token=token,
            resume_download=True,
            max_workers=1,
        )
        print("Downloaded:", model_id)
        last_err=None
        break
    except Exception as e:
        last_err=e
        sleep_s=10*attempt
        print(f"[snapshot_download] attempt {attempt}/7 failed: {e}")
        time.sleep(sleep_s)

if last_err: raise last_err
PY
BASH

COPY server.py /app/server.py

EXPOSE 8000

# Runtime offline (no HTTP calls); relies on baked cache
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

