#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

# This command is an engineering-only synthetic optical-field resource and
# gradient audit. It is deliberately not a formal ImageNet training launcher.
DEPTHS="${DEPTHS:-16,32,64,100}"
FEEDBACK_METHODS="${FEEDBACK_METHODS:-bp_current,fa_source,fa_random}"
FEEDBACK_RANDOM_SEED="${FEEDBACK_RANDOM_SEED:-20260901}"
BATCH_SIZE="${BATCH_SIZE:-1}"
WARMUP_STEPS="${WARMUP_STEPS:-1}"
MEASUREMENT_STEPS="${MEASUREMENT_STEPS:-2}"
ALPHA_MODE="${ALPHA_MODE:-epsilon_probe}"
ALPHA_EPSILON="${ALPHA_EPSILON:-0.01}"
P13_GPU="${P13_GPU:-0}"
OUTPUT_DIRECTORY="${OUTPUT_DIRECTORY:-${EXPERIMENT}/runs/p13_gpu_engineering_sweep_bs1}"
export CUDA_VISIBLE_DEVICES="${P13_GPU}"

RESUME_ARGUMENTS=()
if [[ "${RESUME_EXISTING:-0}" == "1" ]]; then
  RESUME_ARGUMENTS+=(--resume-existing)
fi

if [[ ! -f "${STEM_CHECKPOINT}" ]]; then
  echo "Missing frozen Qwen stem: ${STEM_CHECKPOINT}" >&2
  exit 2
fi
if [[ ! -f "${P11_CHECKPOINT}" ]]; then
  echo "Missing official P11 backbone export: ${P11_CHECKPOINT}" >&2
  exit 2
fi

"${PYTHON_BIN}" -m \
  experiments.qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone.gpu_engineering_sweep \
  --stem-checkpoint "${STEM_CHECKPOINT}" \
  --p11-checkpoint "${P11_CHECKPOINT}" \
  --output-directory "${OUTPUT_DIRECTORY}" \
  --depths "${DEPTHS}" \
  --feedback-methods "${FEEDBACK_METHODS}" \
  --feedback-random-seed "${FEEDBACK_RANDOM_SEED}" \
  --batch-size "${BATCH_SIZE}" \
  --warmup-steps "${WARMUP_STEPS}" \
  --measurement-steps "${MEASUREMENT_STEPS}" \
  --alpha-mode "${ALPHA_MODE}" \
  --alpha-epsilon "${ALPHA_EPSILON}" \
  --phase-learning-rate "${PHASE_LEARNING_RATE:-0.01}" \
  --electronic-learning-rate "${ELECTRONIC_LEARNING_RATE:-0.001}" \
  --seed "${SEED:-2026}" \
  --device cuda:0 \
  --activation-checkpointing \
  "${RESUME_ARGUMENTS[@]}"
