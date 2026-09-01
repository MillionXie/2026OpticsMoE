#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/guest3/miniconda3/envs/xml/bin/python}"
EXPERIMENT="FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_dual_scale_optical_imagenet_backbone"
MODULE="experiments.qwen3_vl_patch_stem_8stage_dual_scale_optical_imagenet_backbone"
FIXED_FEEDBACK_RUNS_ROOT="${FIXED_FEEDBACK_RUNS_ROOT:-FixedFeedbackSFT/runs}"
RUNS_DIR="${FIXED_FEEDBACK_RUNS_ROOT}/qwen3_vl_patch_stem_8stage_dual_scale_optical_imagenet_backbone"
P09_EXPERIMENT="FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone"
STEM_CHECKPOINT="${P09_EXPERIMENT}/assets/qwen3_vl_static_stem_224.pt"

ensure_stem() {
  if [[ ! -f "${STEM_CHECKPOINT}" ]]; then
    bash "${P09_EXPERIMENT}/commands/01_extract_stem.sh"
  fi
}

gpu_uuid() {
  local physical_index="$1"
  nvidia-smi --query-gpu=uuid --format=csv,noheader | sed -n "$((physical_index + 1))p"
}
