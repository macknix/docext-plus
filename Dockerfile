# =============================================================================
# docext_batch - slim GPU image for a Trusted Research Environment (TRE)
# =============================================================================
# Built from an official NVIDIA CUDA *runtime* base and pip-installs the stack
# itself, kept as small as practical for an air-gapped TRE:
#   1. CUDA base pinned to 12.4 — matches the TRE driver (nvidia-smi: CUDA 12.4),
#      no reliance on minor-version forward compatibility.
#   2. NO flash-attn (not in docext's requirements; vLLM falls back to xformers /
#      torch SDPA, and we serve with `--enforce-eager` anyway).
#   3. transformers pinned to docext's tested range (a fresh install otherwise
#      pulls transformers 5.x, which breaks vLLM 0.8.3 with a ProcessorMixin error).
#   4. Build tooling lives only in the builder stage, so it isn't shipped.
# See "How the Dockerfile was built" in the README for the full rationale.
#
# NOTE ON ARCHITECTURE: amd64/x86_64 only (CUDA image). On an Apple Silicon Mac
# build for amd64 to EXPORT to the TRE (it does not run on the Mac):
#   docker build --platform linux/amd64 -t ner-ocr-docext:amd64 .
# Or use scripts/export_image.sh. For local runs use Dockerfile.dev (Ollama).
#
# Build (mount models at run time - recommended):
#   docker build --platform linux/amd64 -t ner-ocr-docext:amd64 .
# Build (bake one model in; needs BuildKit + HF token):
#   DOCKER_BUILDKIT=1 docker buildx build --platform linux/amd64 --load \
#     --secret id=hf_token,env=HUGGING_FACE_HUB_TOKEN \
#     --build-arg BAKE_MODEL=true \
#     --build-arg MODEL_REPO=Qwen/Qwen2.5-VL-7B-Instruct-AWQ \
#     -t ner-ocr-docext:amd64 .
# -----------------------------------------------------------------------------

# CUDA 12.4 runtime to match the TRE driver. `...-base...` (no cuDNN/runtime libs)
# is smaller still and often works because torch bundles its own CUDA libs, but
# `-runtime` is the safe, proven choice; override with --build-arg if you want to
# experiment.
ARG CUDA_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04
ARG DOCEXT_VERSION=0.1.11

# ----------------------------------------------------------------------------
# Stage 1: builder — install the Python stack into a venv (build tools stay here)
# ----------------------------------------------------------------------------
FROM ${CUDA_IMAGE} AS builder
ARG DOCEXT_VERSION
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl build-essential \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

# Isolated venv we can copy wholesale into the runtime stage.
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Installing docext pulls vLLM 0.8.3 -> torch 2.6 (CUDA 12.4 wheels), transformers,
# litellm, pandas, Pillow, etc. It does NOT pull flash-attn (we never add it).
#
# IMPORTANT: docext 0.1.11's PyPI metadata does NOT pin transformers, and vLLM
# 0.8.3 leaves its upper bound open, so a fresh build pulls transformers 5.x —
# which is incompatible (you get "Could not import module 'ProcessorMixin'").
# Pin it to docext's own tested range (its repo requirements.txt: >=4.51.1,<4.53.0).
# PyYAML is the only extra our wrapper needs on top.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir \
        "docext==${DOCEXT_VERSION}" \
        "transformers>=4.51.1,<4.53.0" \
        "PyYAML>=6.0"

# ----------------------------------------------------------------------------
# Stage 2: runtime — CUDA runtime + Python + the venv, no build tools
# ----------------------------------------------------------------------------
FROM ${CUDA_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        poppler-utils libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Application code + example config.
COPY docext_batch/ ./docext_batch/
COPY config.example.yaml ./config.example.yaml

# Models live in a mountable cache by default. Mount your host model dir here.
ENV HF_HOME=/models/hf_cache
RUN mkdir -p /models/hf_cache

# Optionally bake one model into the image (fully offline, no mounts needed).
ARG BAKE_MODEL=false
ARG MODEL_REPO=Qwen/Qwen2.5-VL-7B-Instruct-AWQ
RUN --mount=type=secret,id=hf_token,required=false \
    if [ "$BAKE_MODEL" = "true" ]; then \
        echo "Baking model ${MODEL_REPO} into image..." && \
        ( [ -f /run/secrets/hf_token ] && export HUGGING_FACE_HUB_TOKEN="$(cat /run/secrets/hf_token)" || true ) && \
        python -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL_REPO}', ignore_patterns=['*.pth','original/*','*.gguf'])" ; \
    else \
        echo "Skipping model bake (BAKE_MODEL=$BAKE_MODEL); mount models at /models/hf_cache."; \
    fi

# TRE hardening: no outbound calls at run time.
ENV PYTHONPATH=/app \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    GRADIO_ANALYTICS_ENABLED=False

ENTRYPOINT ["python", "-m", "docext_batch.run"]
CMD ["--config", "/app/config.yaml"]
