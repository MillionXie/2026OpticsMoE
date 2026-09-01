#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_training_common.sh"

select_progressive_growth_stage() {
  local target_depth="$1"
  case "${target_depth}" in
    32)
      PARENT_RUN_NAME="p13_growth16_fa_source_20e_gb192"
      TARGET_RUN_NAME="p13_growth32_fa_source_20e_gb192"
      ;;
    64)
      PARENT_RUN_NAME="p13_growth32_fa_source_20e_gb192"
      TARGET_RUN_NAME="p13_growth64_fa_source_20e_gb192"
      ;;
    100)
      PARENT_RUN_NAME="p13_growth64_fa_source_20e_gb192"
      TARGET_RUN_NAME="p13_growth100_fa_source_20e_gb192"
      ;;
    *)
      echo "TARGET_DEPTH must be exactly 32, 64, or 100; got ${target_depth}." >&2
      return 1
      ;;
  esac
  PARENT_CHECKPOINT="${RUNS_DIR}/${PARENT_RUN_NAME}/checkpoints/best_full_depth.pt"
  TARGET_RUN_DIR="${RUNS_DIR}/${TARGET_RUN_NAME}"
  TARGET_CONFIG="${EXPERIMENT}/configs/generated/${TARGET_RUN_NAME}.yaml"
  TARGET_BASE_LOG="${RUNS_DIR}/logs/${TARGET_RUN_NAME}.log"
  TARGET_LATEST_LOG="${RUNS_DIR}/logs/${TARGET_RUN_NAME}.latest.log"
  TARGET_PID_FILE="${TARGET_RUN_DIR}/launch.pid"
  TARGET_LOCK_FILE="${TARGET_RUN_DIR}/launch.lock"
}

render_or_verify_progressive_config() {
  local target_depth="$1"
  if [[ ! -f "${FORMAL_STEM_CHECKPOINT}" ]]; then
    echo "Required frozen Qwen stem is missing: ${FORMAL_STEM_CHECKPOINT}" >&2
    return 1
  fi
  if [[ ! -f "${PARENT_CHECKPOINT}" ]]; then
    echo "Guarded growth ${target_depth} refuses to render/start before the previous best exists: ${PARENT_CHECKPOINT}" >&2
    return 1
  fi
  "${PYTHON_BIN}" -m \
    experiments.qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone.render_progressive_growth_config \
    --template-config "${EXPERIMENT}/configs/growth16_fa_source_20e_gb192.yaml" \
    --parent-checkpoint "${PARENT_CHECKPOINT}" \
    --target-depth "${target_depth}" \
    --output-config "${TARGET_CONFIG}" \
    --output-dir "${TARGET_RUN_DIR}" \
    --repository "${REPO_DIR}"
}
