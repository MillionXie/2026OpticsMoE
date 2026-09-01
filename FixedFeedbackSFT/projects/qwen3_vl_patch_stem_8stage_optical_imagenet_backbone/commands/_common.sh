#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/guest3/miniconda3/envs/xml/bin/python}"
EXPERIMENT="FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone"
MODULE="experiments.qwen3_vl_patch_stem_8stage_optical_imagenet_backbone"
FIXED_FEEDBACK_RUNS_ROOT="${FIXED_FEEDBACK_RUNS_ROOT:-FixedFeedbackSFT/runs}"
RUNS_DIR="${FIXED_FEEDBACK_RUNS_ROOT}/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone"
STEM_CHECKPOINT="${EXPERIMENT}/assets/qwen3_vl_static_stem_224.pt"

select_gpu() {
  : "${PHYSICAL_GPU_INDEX:?Set PHYSICAL_GPU_INDEX to an nvidia-smi index}"
  local uuid
  uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | sed -n "$((PHYSICAL_GPU_INDEX + 1))p")"
  : "${uuid:?Could not resolve GPU ${PHYSICAL_GPU_INDEX}}"
  export CUDA_VISIBLE_DEVICES="${uuid}"
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
}

ensure_stem() {
  if [[ ! -f "${STEM_CHECKPOINT}" ]]; then
    bash "${SCRIPT_DIR}/01_extract_stem.sh"
  fi
}
