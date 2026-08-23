#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

RUN_DIR="${EXPERIMENT}/runs/p09_imagenet1k_pretrain_bs96_90e"
LOG="${EXPERIMENT}/logs/p09_imagenet1k_pretrain_bs96_90e.log"
echo "PID file: $(cat "${RUN_DIR}/launch.pid" 2>/dev/null || echo missing)"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
if [[ -f "${RUN_DIR}/metrics/latest.json" ]]; then
  "${PYTHON_BIN}" - "${RUN_DIR}/metrics/latest.json" <<'PY'
import json
import sys

row = json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps({
    "epoch": row["epoch"],
    "train_top1": row["train"]["top1_accuracy"],
    "validation_top1": row["validation"]["top1_accuracy"],
    "phase_motion_rad": row["phase_motion"]["mean_absolute_rad"],
    "optical_gates": row["optical_gates"],
    "electronic_skip_gates": row.get("electronic_skip_gates"),
}, indent=2))
fi
tail -n 40 "${LOG}" 2>/dev/null || true
