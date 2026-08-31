#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_training_common.sh"

RUN_DIR="${EXPERIMENT}/runs/p11_epoch88_matched_continue_20e_gb192"
LOG="${EXPERIMENT}/logs/p11_epoch88_matched_continue_20e_gb192.latest.log"
LOCK_FILE="${RUN_DIR}/launch.lock"
pid="$(cat "${RUN_DIR}/launch.pid" 2>/dev/null || true)"
if [[ -f "${LOCK_FILE}" ]] && ! flock -n "${LOCK_FILE}" true 2>/dev/null; then
  echo "PID=${pid} state=running"
else
  echo "PID=${pid:-not-started} state=not-running"
fi
echo "latest_log=${LOG}"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
if [[ -f "${RUN_DIR}/metrics/latest.json" ]]; then
  "${PYTHON_BIN}" -m json.tool "${RUN_DIR}/metrics/latest.json"
fi
if [[ -f "${RUN_DIR}/result.json" ]]; then
  "${PYTHON_BIN}" -m json.tool "${RUN_DIR}/result.json"
fi
tail -n 40 "${LOG}" 2>/dev/null || true
