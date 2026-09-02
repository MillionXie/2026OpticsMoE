#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPOSITORY_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/guest3/miniconda3/envs/xml/bin/python}"
RUN_DIR="FixedFeedbackSFT/runs/qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone/p11_large_recipe_formal_100e_phase7e3_5gpu_gb480"
HISTORY="${RUN_DIR}/metrics/history.json"
OUTPUT="${RUN_DIR}/metrics/early_gate_latest.json"

if [[ ! -f "${HISTORY}" ]]; then
  echo "No completed formal epoch history yet: ${HISTORY}" >&2
  exit 1
fi

exec "${PYTHON_BIN}" FixedFeedbackSFT/tools/assess_p11_longrun_gate.py \
  --history "${HISTORY}" --output "${OUTPUT}"
