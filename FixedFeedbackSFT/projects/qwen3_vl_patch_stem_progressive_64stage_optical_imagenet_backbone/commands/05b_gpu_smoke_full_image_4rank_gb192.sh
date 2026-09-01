#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_training_common.sh"
require_training_sources

PHYSICAL_GPU_INDICES="${PHYSICAL_GPU_INDICES:?Set exactly four free physical GPU indices, e.g. 0,1,3,4}"
P13_ACTION="${P13_ACTION:-fresh}"
IFS=',' read -r -a indices <<< "${PHYSICAL_GPU_INDICES}"
if [[ "${#indices[@]}" -ne 4 ]]; then
  echo "The four-rank P13 smoke requires exactly four GPUs." >&2
  exit 1
fi

CONFIG="${EXPERIMENT}/configs/gpu_smoke_full_image_4rank_gb192.yaml"
RUN_DIR="${RUNS_DIR}/p13_growth16_full_image_4rank_gb192_gpu_smoke"
acquire_launch_lock "${RUN_DIR}/launch.lock"
MODE="$(training_mode_argument "${P13_ACTION}" "${RUN_DIR}")"
export CUDA_VISIBLE_DEVICES="$(visible_gpu_uuids "${PHYSICAL_GPU_INDICES}")"
TORCHRUN_BIN="$(dirname "${PYTHON_BIN}")/torchrun"

echo "Running foreground P13 four-rank smoke on physical GPUs ${PHYSICAL_GPU_INDICES}; effective_global_batch=192"
exec "${TORCHRUN_BIN}" --standalone --nproc_per_node=4 \
  -m "${MODULE}.train" --config "${CONFIG}" "${MODE}" 9>&9
