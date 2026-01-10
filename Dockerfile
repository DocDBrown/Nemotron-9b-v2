FROM dustynv/l4t-pytorch:r36.4.0

ARG DEBIAN_FRONTEND=noninteractive

# Use PyPI for general packages; Jetson wheel index only when explicitly used
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    PIP_INDEX_URL=https://pypi.org/simple \
    # Bake HF cache into the image here:
    HF_HOME=/opt/hf \
    HF_HUB_CACHE=/opt/hf/hub \
    TORCH_HOME=/opt/torch \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

ARG JETSON_WHEELS=https://pypi.jetson-ai-lab.io/jp6/cu126

# Optional: allow pulling private/gated repos during build
# WARNING: if you pass this as a build-arg, it can end up in image history.
ARG HF_TOKEN=""

RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential cmake ninja-build pkg-config python3-dev \
      libopenblas-dev libffi-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install -U pip setuptools wheel

# Model/runtime deps from PyPI
RUN python3 -m pip install \
      "transformers==4.48.3" \
      "accelerate>=0.30.0" \
      safetensors sentencepiece huggingface_hub

# Web server deps
RUN python3 -m pip install fastapi uvicorn

# Quantization (try Jetson wheels first, fall back)
RUN python3 -m pip install --extra-index-url ${JETSON_WHEELS} bitsandbytes || \
    python3 -m pip install bitsandbytes || true

# Nemotron depends on mamba-ssm; try Jetson wheels first; fall back to build.
RUN set -eux; \
    if python3 -c "import mamba_ssm" 2>/dev/null; then \
      echo "mamba_ssm already present"; \
    else \
      (python3 -m pip install --extra-index-url ${JETSON_WHEELS} "mamba-ssm[causal-conv1d]" --no-build-isolation) || ( \
        echo "Jetson wheel install failed; attempting source build (may require nvcc)"; \
        apt-get update; \
        (apt-get install -y --no-install-recommends cuda-nvcc-12-6 cuda-toolkit-12-6) || true; \
        rm -rf /var/lib/apt/lists/*; \
        python3 -m pip install "mamba-ssm[causal-conv1d]" --no-build-isolation \
      ); \
    fi

# Optional: triton (not fatal)
RUN python3 -m pip install --extra-index-url ${JETSON_WHEELS} triton || \
    python3 -m pip install triton || true

# ----------------------------
# PRE-DOWNLOAD MODEL AT BUILD
# ----------------------------
# snapshot_download will populate the HF cache under HF_HOME/HF_HUB_CACHE.
# This is the supported way to prepare offline usage. :contentReference[oaicite:1]{index=1}
RUN set -eux; \
    python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

model_id = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"

token = os.environ.get("HF_TOKEN") or None
# Download full repo into the Hugging Face cache inside the image
snapshot_download(
    repo_id=model_id,
    token=token,
)
print("Downloaded:", model_id)
PY

WORKDIR /opt
COPY server.py /opt/server.py

EXPOSE 8000

# Force offline at runtime: no HTTP calls; only cached files are used. :contentReference[oaicite:2]{index=2}
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
