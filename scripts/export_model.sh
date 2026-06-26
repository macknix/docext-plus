#!/usr/bin/env bash
# =============================================================================
# Tar up the local model cache so it can be moved into a network-isolated TRE.
# =============================================================================
# Run on your networked machine after ./scripts/download_model.sh has populated
# ./models/hf_cache. Produces models.tar.gz containing the whole models/ tree;
# extract it in the TRE working directory and mount models/hf_cache at run time.
#
# Usage:
#   ./scripts/download_model.sh Qwen/Qwen2.5-VL-7B-Instruct-AWQ
#   ./scripts/export_model.sh                 # -> models.tar.gz
#
# In the TRE:
#   tar -xzf models.tar.gz                    # recreates ./models/hf_cache/...
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${OUT:-models.tar.gz}"

if [ ! -d "$ROOT/models/hf_cache" ]; then
    echo "ERROR: $ROOT/models/hf_cache not found. Run scripts/download_model.sh first." >&2
    exit 1
fi

echo ">> Archiving models/ -> $OUT (verbose: listing each file as it is added)"
# -v lists files as they are archived; -C "$ROOT" stores paths as models/hf_cache/...
tar -czvf "$OUT" -C "$ROOT" models

echo ">> Done: $OUT ($(du -h "$OUT" | cut -f1))"
echo "   In the TRE:  tar -xzf $OUT   (then mount ./models/hf_cache at /models/hf_cache)"
