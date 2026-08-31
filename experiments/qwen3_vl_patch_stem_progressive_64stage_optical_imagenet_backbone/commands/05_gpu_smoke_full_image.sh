#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_training_common.sh"
require_training_sources

P13_GPU="${P13_GPU:?Set one free physical GPU index}"
P13_ACTION="${P13_ACTION:-fresh}"
CONFIG="${EXPERIMENT}/configs/gpu_smoke_full_image.yaml"
RUN_DIR="${EXPERIMENT}/runs/p13_growth16_full_image_gpu_smoke"
acquire_launch_lock "${RUN_DIR}/launch.lock"
MODE="$(training_mode_argument "${P13_ACTION}" "${RUN_DIR}")"
export CUDA_VISIBLE_DEVICES="$(visible_gpu_uuids "${P13_GPU}")"

"${PYTHON_BIN}" -m "${EXPERIMENT//\//.}.train" \
  --config "${CONFIG}" "${MODE}"
