#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

QWEN_CACHE_ROOT="${QWEN_CACHE_ROOT:-/DATA/DATA1/guest3/.cache/huggingface/hub/models--Qwen--Qwen3-VL-Embedding-2B}"
QWEN_SAFETENSORS="${QWEN_SAFETENSORS:-$(find "${QWEN_CACHE_ROOT}/snapshots" -mindepth 2 -maxdepth 2 -name model.safetensors -print -quit)}"
: "${QWEN_SAFETENSORS:?Qwen3-VL-Embedding-2B model.safetensors was not found}"

"${PYTHON_BIN}" -m "${EXPERIMENT//\//.}.extract_stem" \
  --checkpoint "${QWEN_SAFETENSORS}" \
  --output "${STEM_CHECKPOINT}" \
  --image-size 224
