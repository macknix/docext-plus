#!/usr/bin/env bash
# =============================================================================
# Pre-download a Hugging Face model into ./models/hf_cache for offline / TRE use.
# =============================================================================
# Run this on a networked machine, then transport the whole ./models directory
# into the TRE (e.g. tar it up) and mount it at /models/hf_cache when running the
# GPU image. The container runs with HF_HUB_OFFLINE=1, so the model must already
# be present in the mounted cache.
#
# IMPORTANT: docext needs a VISION-language model (it sends images). Use a VLM,
# e.g.  Qwen/Qwen2.5-VL-7B-Instruct-AWQ  (default)  - NOT a text-only model.
#
# Usage:
#   ./scripts/download_model.sh Qwen/Qwen2.5-VL-7B-Instruct-AWQ
#   HUGGING_FACE_HUB_TOKEN=hf_xxx ./scripts/download_model.sh <repo>   # gated models
#
# Requires: pip install huggingface_hub   (any 0.x or 1.x version)
set -euo pipefail

MODEL_REPO="${1:-Qwen/Qwen2.5-VL-7B-Instruct-AWQ}"
CACHE_DIR="${HF_CACHE_DIR:-$(cd "$(dirname "$0")/.." && pwd)/models/hf_cache}"

mkdir -p "$CACHE_DIR"
echo "Downloading '$MODEL_REPO' into $CACHE_DIR ..."

# Use the Python API directly so this works with any huggingface_hub version
# (the CLI was renamed from `huggingface-cli` to `hf` in huggingface_hub 1.x).
HF_HOME="$CACHE_DIR" python - "$MODEL_REPO" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo = sys.argv[1]
path = snapshot_download(
    repo_id=repo,
    # Skip redundant weight formats some repos ship; harmless if absent.
    ignore_patterns=["*.pth", "original/*", "*.gguf"],
)
print(f"Downloaded to: {path}")
PY

echo "Done. Mount it at run time with:"
echo "  -v \"$CACHE_DIR\":/models/hf_cache"
echo "and select it with: --model hosted_vllm/$MODEL_REPO"
