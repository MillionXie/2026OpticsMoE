#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/guest3/miniconda3/envs/xml/bin/python}"
EXPERIMENT="FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone"
MODULE="experiments.qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone"
FIXED_FEEDBACK_RUNS_ROOT="${FIXED_FEEDBACK_RUNS_ROOT:-FixedFeedbackSFT/runs}"
RUNS_DIR="${FIXED_FEEDBACK_RUNS_ROOT}/qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone"
P09_EXPERIMENT="FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone"
STEM_CHECKPOINT="${P09_EXPERIMENT}/assets/qwen3_vl_static_stem_224.pt"
FROZEN_P11_ASSET="FixedFeedbackSFT/runs/_assets/8stage"

ensure_stem() {
  if [[ ! -f "${STEM_CHECKPOINT}" ]]; then
    bash "${P09_EXPERIMENT}/commands/01_extract_stem.sh"
  fi
}

ensure_frozen_p11_asset() {
  local required
  for required in \
    "${FROZEN_P11_ASSET}/checkpoints/backbone.pt" \
    "${FROZEN_P11_ASSET}/checkpoints/best.pt" \
    "${FROZEN_P11_ASSET}/dependencies/qwen3_vl_static_stem_224.pt" \
    "${FROZEN_P11_ASSET}/manifest.json"; do
    if [[ ! -f "${required}" ]]; then
      echo "Missing immutable P11 asset: ${required}" >&2
      echo "Run FixedFeedbackSFT/commands/00_freeze_8_16_backbones.sh first." >&2
      return 1
    fi
  done
}

gpu_uuid() {
  local physical_index="$1"
  [[ "${physical_index}" =~ ^[0-9]+$ ]] || {
    echo "Invalid physical GPU index: ${physical_index}" >&2
    return 1
  }
  local uuid
  uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | sed -n "$((physical_index + 1))p")"
  [[ -n "${uuid}" ]] || {
    echo "Physical GPU index ${physical_index} does not exist." >&2
    return 1
  }
  printf '%s\n' "${uuid}"
}

require_idle_gpu() {
  local physical_index="$1"
  gpu_uuid "${physical_index}" >/dev/null
  local row memory_mib utilization
  row="$(nvidia-smi -i "${physical_index}" \
    --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)"
  IFS=',' read -r memory_mib utilization <<< "${row}"
  memory_mib="${memory_mib//[[:space:]]/}"
  utilization="${utilization//[[:space:]]/}"
  [[ "${memory_mib}" =~ ^[0-9]+$ && "${utilization}" =~ ^[0-9]+$ ]] || {
    echo "Could not parse GPU ${physical_index} occupancy: ${row}" >&2
    return 1
  }
  local max_memory="${P11_IDLE_MAX_MEMORY_MIB:-512}"
  local max_utilization="${P11_IDLE_MAX_UTILIZATION_PERCENT:-20}"
  if (( memory_mib > max_memory || utilization > max_utilization )); then
    echo "GPU ${physical_index} is not idle: memory=${memory_mib} MiB, utilization=${utilization}%." >&2
    return 1
  fi
}
