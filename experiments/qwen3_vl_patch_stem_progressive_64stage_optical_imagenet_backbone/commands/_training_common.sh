#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

FORMAL_STEM_CHECKPOINT="experiments/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/assets/qwen3_vl_static_stem_224.pt"
FORMAL_P11_BACKBONE_CHECKPOINT="${P11_EXPERIMENT}/runs/p11_imagenet1k_pretrain_bs96_90e/checkpoints/backbone.pt"
FORMAL_P11_TRAINING_CHECKPOINT="${P11_EXPERIMENT}/runs/p11_imagenet1k_pretrain_bs96_90e/checkpoints/best.pt"

require_training_sources() {
  local path
  for path in \
    "${FORMAL_STEM_CHECKPOINT}" \
    "${FORMAL_P11_BACKBONE_CHECKPOINT}" \
    "${FORMAL_P11_TRAINING_CHECKPOINT}"; do
    if [[ ! -f "${path}" ]]; then
      echo "Required training source is missing: ${path}" >&2
      return 1
    fi
  done
}

gpu_uuid() {
  local physical_index="$1"
  nvidia-smi --query-gpu=uuid --format=csv,noheader | sed -n "$((physical_index + 1))p"
}

visible_gpu_uuids() {
  local comma_indices="$1"
  local indices=()
  local uuids=()
  declare -A seen=()
  IFS=',' read -r -a indices <<< "${comma_indices}"
  for index in "${indices[@]}"; do
    if [[ ! "${index}" =~ ^[0-9]+$ ]]; then
      echo "GPU index must be a non-negative integer, got ${index}." >&2
      return 1
    fi
    if [[ -n "${seen[${index}]+present}" ]]; then
      echo "GPU index ${index} was specified more than once." >&2
      return 1
    fi
    seen[${index}]=1
    local uuid
    uuid="$(gpu_uuid "${index}")"
    uuid="${uuid//$'\r'/}"
    uuid="${uuid//[[:space:]]/}"
    if [[ -z "${uuid}" ]]; then
      echo "No GPU UUID found for physical index ${index}." >&2
      return 1
    fi
    uuids+=("${uuid}")
  done
  (IFS=','; echo "${uuids[*]}")
}

training_run_has_artifacts() {
  local run_dir="$1"
  local path
  for path in \
    "${run_dir}/manifest.json" \
    "${run_dir}/manifest.json.tmp" \
    "${run_dir}/initial_phases.pt" \
    "${run_dir}/initial_phases.pt.tmp" \
    "${run_dir}/result.json" \
    "${run_dir}/result.json.tmp"; do
    if [[ -e "${path}" ]]; then
      return 0
    fi
  done
  local directory
  for directory in "${run_dir}/checkpoints" "${run_dir}/metrics"; do
    if [[ -d "${directory}" ]] &&
      find "${directory}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
      return 0
    fi
  done
  return 1
}

training_mode_argument() {
  local action="$1"
  local run_dir="$2"
  case "${action}" in
    fresh)
      if training_run_has_artifacts "${run_dir}"; then
        echo "Fresh mode refuses existing run artifacts in ${run_dir}" >&2
        return 1
      fi
      echo "--fresh"
      ;;
    resume)
      if [[ ! -f "${run_dir}/checkpoints/last.pt" ]]; then
        echo "Resume mode requires ${run_dir}/checkpoints/last.pt" >&2
        return 1
      fi
      echo "--resume"
      ;;
    *)
      echo "Action must be exactly fresh or resume, got ${action}." >&2
      return 1
      ;;
  esac
}

acquire_launch_lock() {
  local lock_file="$1"
  if ! command -v flock >/dev/null 2>&1; then
    echo "flock is required for race-free training launch." >&2
    return 1
  fi
  mkdir -p "$(dirname "${lock_file}")"
  exec 9>"${lock_file}"
  if ! flock -n 9; then
    echo "Another launcher/training process holds ${lock_file}." >&2
    return 1
  fi
}

segmented_log_path() {
  local base_log="$1"
  local action="$2"
  local timestamp
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  printf '%s.%s.%s.%s.log\n' "${base_log%.log}" "${timestamp}" "${action}" "$$"
}

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
