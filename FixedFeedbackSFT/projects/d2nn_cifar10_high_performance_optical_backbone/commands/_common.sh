#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/guest3/miniconda3/envs/xml/bin/python}"

select_gpu() {
  : "${PHYSICAL_GPU_INDEX:?Set PHYSICAL_GPU_INDEX to the nvidia-smi GPU index}"
  local gpu_uuid
  gpu_uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | sed -n "$((PHYSICAL_GPU_INDEX + 1))p")"
  : "${gpu_uuid:?Could not resolve PHYSICAL_GPU_INDEX=${PHYSICAL_GPU_INDEX}}"
  export CUDA_VISIBLE_DEVICES="${gpu_uuid}"
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
}
