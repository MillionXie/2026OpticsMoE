#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

OUTPUT_DIRECTORY="${OUTPUT_DIRECTORY:-${EXPERIMENT}/runs/p13_migrated_64stage_initialization}"

if [[ ! -f "${STEM_CHECKPOINT}" ]]; then
  echo "Missing frozen Qwen stem: ${STEM_CHECKPOINT}" >&2
  exit 2
fi
if [[ ! -f "${P11_CHECKPOINT}" ]]; then
  echo "Missing official P11 backbone export: ${P11_CHECKPOINT}" >&2
  exit 2
fi

"${PYTHON_BIN}" -m \
  experiments.qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone.migration \
  --stem-checkpoint "${STEM_CHECKPOINT}" \
  --p11-checkpoint "${P11_CHECKPOINT}" \
  --output-directory "${OUTPUT_DIRECTORY}" \
  --num-stages 64 \
  --new-stage-alpha-init 0 \
  --new-stage-alpha-epsilon 0.01 \
  --new-stage-ramp-epochs 10 \
  --activation-checkpointing

