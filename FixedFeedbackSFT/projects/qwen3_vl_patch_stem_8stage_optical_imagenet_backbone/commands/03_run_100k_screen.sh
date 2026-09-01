#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
ensure_stem
PHYSICAL_GPU_INDEX="${PHYSICAL_GPU_INDEX:-3}"
select_gpu
"${PYTHON_BIN}" -m "${MODULE}.train" \
  --config "${EXPERIMENT}/configs/screen_100k.yaml" --resume
