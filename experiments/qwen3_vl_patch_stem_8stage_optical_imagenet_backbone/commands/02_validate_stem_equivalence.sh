#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
ensure_stem

QWEN_CACHE_ROOT="${QWEN_CACHE_ROOT:-/DATA/DATA1/guest3/.cache/huggingface/hub/models--Qwen--Qwen3-VL-Embedding-2B}"
QWEN_DIRECTORY="${QWEN_DIRECTORY:-$(find "${QWEN_CACHE_ROOT}/snapshots" -mindepth 1 -maxdepth 1 -type d -print -quit)}"
: "${QWEN_DIRECTORY:?Qwen3-VL-Embedding-2B snapshot was not found}"

"${PYTHON_BIN}" -m "${EXPERIMENT//\//.}.validate_stem" \
  --qwen-directory "${QWEN_DIRECTORY}" \
  --stem-checkpoint "${STEM_CHECKPOINT}" \
  --report "${EXPERIMENT}/assets/qwen3_vl_static_stem_224_equivalence.json"
