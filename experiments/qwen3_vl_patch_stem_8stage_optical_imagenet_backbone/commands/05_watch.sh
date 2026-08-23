#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

RUN="${1:-${EXPERIMENT}/runs/p08_imagenet1k_pretrain_90e}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
ps -u "$(id -un)" -o pid,etimes,cmd | grep -E 'qwen3_vl_patch_stem_8stage|torchrun' | grep -v grep || true
if [[ -f "${RUN}/metrics/latest.json" ]]; then
  "${PYTHON_BIN}" -m json.tool "${RUN}/metrics/latest.json"
elif [[ -f "${RUN}/metrics/initial_baseline.json" ]]; then
  "${PYTHON_BIN}" -m json.tool "${RUN}/metrics/initial_baseline.json"
else
  echo "No metrics have been written under ${RUN} yet."
fi
