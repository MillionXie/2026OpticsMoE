#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54"
RUN_DIR=""
INTERVAL=0
while (($#)); do
  case "$1" in
    --run-dir) RUN_DIR="${2:?}"; shift 2 ;;
    --interval) INTERVAL="${2:?}"; shift 2 ;;
    *) echo "Usage: $0 [--run-dir DIR] [--interval SECONDS]" >&2; exit 2 ;;
  esac
done
if [[ -z "${RUN_DIR}" ]]; then RUN_DIR="$(head -n 1 "${PROJECT}/runs/server_jobs/latest_run.txt")"; fi
[[ -d "${RUN_DIR}" ]] || { echo "Missing run directory: ${RUN_DIR}" >&2; exit 2; }

snapshot() {
  date
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader || true
  for name in spatial temporal temporal_accuracy; do
    printf '\n[%s]\n' "${name}"
    if [[ -f "${RUN_DIR}/${name}.pid" ]]; then
      pid="$(cat "${RUN_DIR}/${name}.pid")"
      if kill -0 "${pid}" 2>/dev/null; then echo "RUNNING PID=${pid}"; else echo "EXITED PID=${pid}"; fi
      tail -n 12 "${RUN_DIR}/${name}.log" 2>/dev/null || true
    else
      echo "not launched"
    fi
  done
}
while true; do snapshot; (( INTERVAL > 0 )) || break; sleep "${INTERVAL}"; done
