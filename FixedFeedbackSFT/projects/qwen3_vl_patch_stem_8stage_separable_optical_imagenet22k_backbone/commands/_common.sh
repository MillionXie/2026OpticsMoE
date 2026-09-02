#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/guest3/miniconda3/envs/xml/bin/python}"
MODULE="experiments.qwen3_vl_patch_stem_8stage_separable_optical_imagenet22k_backbone.train"
DATASET_MODULE="experiments.qwen3_vl_patch_stem_8stage_separable_optical_imagenet22k_backbone.dataset"
PROJECT="FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_separable_optical_imagenet22k_backbone"
RUNS_ROOT="${FIXED_FEEDBACK_RUNS_ROOT:-FixedFeedbackSFT/runs}"
RUNS_DIR="${RUNS_ROOT}/qwen3_vl_patch_stem_8stage_separable_optical_imagenet22k_backbone"
LOG_DIR="${RUNS_DIR}/_logs"

gpu_uuid() {
  local physical_index="$1"
  [[ "${physical_index}" =~ ^[0-9]+$ ]] || {
    echo "Invalid physical GPU index: ${physical_index}" >&2
    return 1
  }
  local uuid
  uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | sed -n "$((physical_index + 1))p")"
  [[ -n "${uuid}" ]] || {
    echo "GPU ${physical_index} does not exist" >&2
    return 1
  }
  printf '%s\n' "${uuid}"
}

require_idle_gpu() {
  local physical_index="$1"
  local row memory_mib utilization
  row="$(nvidia-smi -i "${physical_index}" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)"
  IFS=',' read -r memory_mib utilization <<< "${row}"
  memory_mib="${memory_mib//[[:space:]]/}"
  utilization="${utilization//[[:space:]]/}"
  local max_memory="${IN22K_IDLE_MAX_MEMORY_MIB:-512}"
  local max_utilization="${IN22K_IDLE_MAX_UTILIZATION_PERCENT:-20}"
  if (( memory_mib > max_memory || utilization > max_utilization )); then
    echo "GPU ${physical_index} is not idle: ${memory_mib} MiB, ${utilization}%" >&2
    return 1
  fi
}

recipe_config() {
  case "${LARGE_DATA_RECIPE:-fall11_full}" in
    fall11_full)
      printf '%s\n' "${PROJECT}/configs/imagenet22k_fall11_21841_90e.yaml"
      ;;
    miil_p_fall11)
      printf '%s\n' "${PROJECT}/configs/imagenet21k_p_fall11_11221_80e.yaml"
      ;;
    *)
      echo "LARGE_DATA_RECIPE must be fall11_full or miil_p_fall11" >&2
      return 1
      ;;
  esac
}

preflight_cpu() {
  local config="$1"
  # This command does not initialize CUDA/NCCL or create an output directory.
  # It intentionally re-hashes the large sample/offset files: formal launch
  # must never trust a fast metadata-only audit.
  "${PYTHON_BIN}" -m "${MODULE}" --config "${config}" --preflight-only \
    --verify-large-index-files
}
