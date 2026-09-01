#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
ensure_stem
ensure_frozen_p11_asset

PHYSICAL_GPU_INDEX="${PHYSICAL_GPU_INDEX:?Set one idle physical GPU index}"
require_idle_gpu "${PHYSICAL_GPU_INDEX}"
export CUDA_VISIBLE_DEVICES="$(gpu_uuid "${PHYSICAL_GPU_INDEX}")"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
CONFIG="${EXPERIMENT}/configs/large_scale_continue_smoke.yaml"
RUN_DIR="${RUNS_DIR}/smoke_large_scale_continue"

if [[ -e "${RUN_DIR}/manifest.json" || -d "${RUN_DIR}/checkpoints" ]]; then
  echo "Smoke output already exists; move it inside FixedFeedbackSFT/runs/_quarantine first." >&2
  exit 1
fi
"${PYTHON_BIN}" -m "${MODULE}.large_scale_continue" --config "${CONFIG}" --fresh
