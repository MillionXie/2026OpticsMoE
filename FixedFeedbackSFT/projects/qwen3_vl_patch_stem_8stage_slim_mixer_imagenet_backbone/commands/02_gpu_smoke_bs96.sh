#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
ensure_stem

PHYSICAL_GPU_INDEX="${PHYSICAL_GPU_INDEX:-5}"
export CUDA_VISIBLE_DEVICES="$(gpu_uuid "${PHYSICAL_GPU_INDEX}")"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

rm -rf "${RUNS_DIR}/gpu_smoke_bs96"
"${PYTHON_BIN}" -m "${MODULE}.train" \
  --config "${EXPERIMENT}/configs/gpu_smoke_bs96.yaml"
