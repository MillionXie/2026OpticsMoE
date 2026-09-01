#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
RUN_DIR="${RUNS_DIR}/p10_imagenet1k_pretrain_bs96_90e"
LOG="${RUNS_DIR}/logs/p10_imagenet1k_pretrain_bs96_90e.log"
echo "PID: $(cat "${RUN_DIR}/launch.pid" 2>/dev/null || echo not-started)"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
tail -n 40 "${LOG}" 2>/dev/null || true
