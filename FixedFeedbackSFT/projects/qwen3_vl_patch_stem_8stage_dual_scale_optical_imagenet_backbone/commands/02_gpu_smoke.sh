#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
ensure_stem

PHYSICAL_GPU_INDEX="${PHYSICAL_GPU_INDEX:-1}"
export CUDA_VISIBLE_DEVICES="$(gpu_uuid "${PHYSICAL_GPU_INDEX}")"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${RUNS_DIR}/gpu_smoke_${RUN_TAG}"
CONFIG_PATH="/tmp/p10_gpu_smoke_${RUN_TAG}.yaml"
sed "s|^output_dir:.*|output_dir: ${OUTPUT_DIR}|" \
  "${EXPERIMENT}/configs/gpu_smoke.yaml" > "${CONFIG_PATH}"

"${PYTHON_BIN}" -m "${MODULE}.train" --config "${CONFIG_PATH}"
echo "P10 smoke result: ${OUTPUT_DIR}/result.json"
