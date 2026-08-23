#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
ensure_stem

PHYSICAL_GPU_INDICES="${PHYSICAL_GPU_INDICES:-3,5}"
IFS=',' read -r -a indices <<< "${PHYSICAL_GPU_INDICES}"
uuids=()
for index in "${indices[@]}"; do
  uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | sed -n "$((index + 1))p")"
  : "${uuid:?Could not resolve GPU ${index}}"
  uuids+=("${uuid}")
done
export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${uuids[*]}")"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
TORCHRUN_BIN="$(dirname "${PYTHON_BIN}")/torchrun"

"${TORCHRUN_BIN}" --standalone --nproc_per_node="${#indices[@]}" \
  -m "${EXPERIMENT//\//.}.train" \
  --config "${EXPERIMENT}/configs/pretrain_90e.yaml" --resume
