#!/usr/bin/env bash
# =============================================================================
# Build the GPU/TRE image for linux/amd64 and export it as a tarball.
# =============================================================================
# The TRE is x86_64 (amd64) but your Mac is ARM64 (Apple Silicon), so we build for
# linux/amd64 (emulated via QEMU on Apple Silicon - slow but fine, since we only
# build+export here; the image runs in the TRE, not on the Mac).
#
# Usage:
#   ./scripts/export_image.sh                       # build + save the image
#   BAKE_MODEL=true HUGGING_FACE_HUB_TOKEN=hf_xxx \
#     MODEL_REPO=Qwen/Qwen2.5-VL-7B-Instruct-AWQ \
#     ./scripts/export_image.sh                     # fully self-contained image
#
# Then move the produced .tar.gz into the TRE and:
#   docker load -i ner-ocr-docext-amd64.tar.gz       # or: gzip -dc ... | docker load
set -euo pipefail

IMAGE="${IMAGE:-ner-ocr-docext:amd64}"
OUT="${OUT:-ner-ocr-docext-amd64.tar.gz}"
PLATFORM="${PLATFORM:-linux/amd64}"
BAKE_MODEL="${BAKE_MODEL:-false}"
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen2.5-VL-7B-Instruct-AWQ}"

echo ">> Building $IMAGE for $PLATFORM (BAKE_MODEL=$BAKE_MODEL)"

if [ "$BAKE_MODEL" = "true" ]; then
    : "${HUGGING_FACE_HUB_TOKEN:?Set HUGGING_FACE_HUB_TOKEN to bake a (possibly gated) model}"
    # buildx is required for --secret; --load puts the result in the local daemon.
    DOCKER_BUILDKIT=1 docker buildx build \
        --platform "$PLATFORM" \
        --secret id=hf_token,env=HUGGING_FACE_HUB_TOKEN \
        --build-arg BAKE_MODEL=true \
        --build-arg MODEL_REPO="$MODEL_REPO" \
        -t "$IMAGE" --load .
else
    docker build --platform "$PLATFORM" -t "$IMAGE" .
fi

echo ">> Saving $IMAGE -> $OUT (this is a multi-GB image; please wait)"
# Show a live progress bar via `pv` if it is installed (brew install pv), else
# fall back to a plain save with no progress output.
if command -v pv >/dev/null 2>&1; then
    docker save "$IMAGE" | pv | gzip > "$OUT"
else
    echo "   (tip: 'brew install pv' for a live progress bar)"
    docker save "$IMAGE" | gzip > "$OUT"
fi

echo ">> Done: $OUT ($(du -h "$OUT" | cut -f1))"
echo "   In the TRE:  docker load -i $OUT   (or podman load -i $OUT)"
