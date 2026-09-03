#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
JOB_ROOT="${PROJECT_DIR}/runs/server_jobs"
RUN_DIR=""
INTERVAL=0
TAIL_LINES=18

usage() {
  cat <<'EOF'
Usage:
  bash monitor_four_runs.sh [--run-dir PATH] [--interval SECONDS] [--tail-lines N]

Without --run-dir, the path in runs/server_jobs/latest_run.txt is used.
The default is a single snapshot.  Set --interval 30 for a repeating monitor;
press Ctrl+C to stop monitoring (the training jobs are not stopped).
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

while (($#)); do
  case "$1" in
    --run-dir) RUN_DIR="${2:?missing value for --run-dir}"; shift 2 ;;
    --interval) INTERVAL="${2:?missing value for --interval}"; shift 2 ;;
    --tail-lines) TAIL_LINES="${2:?missing value for --tail-lines}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "${INTERVAL}" =~ ^[0-9]+$ ]] || die "--interval must be a non-negative integer"
[[ "${TAIL_LINES}" =~ ^[0-9]+$ ]] || die "--tail-lines must be a non-negative integer"

if [[ -z "${RUN_DIR}" ]]; then
  [[ -f "${JOB_ROOT}/latest_run.txt" ]] || die "no latest run record under ${JOB_ROOT}"
  RUN_DIR="$(head -n 1 "${JOB_ROOT}/latest_run.txt")"
fi
[[ -d "${RUN_DIR}" ]] || die "run directory does not exist: ${RUN_DIR}"

snapshot() {
  printf '\n===== %s | %s =====\n' "$(date '+%F %T %Z')" "${RUN_DIR}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu \
      --format=csv,noheader,nounits || true
  fi

  local name pid pid_file state process_args
  for name in spatial spatial_seed43 spatial_strong_robust temporal; do
    pid_file="${RUN_DIR}/${name}.pid"
    if [[ ! -f "${pid_file}" ]]; then
      printf '\n[%s] NOT LAUNCHED (missing PID file)\n' "${name}"
      continue
    fi
    pid="$(tr -cd '0-9' < "${pid_file}" || true)"
    state="EXITED"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      process_args="$(ps -p "${pid}" -o args= 2>/dev/null || true)"
      if [[ "${process_args}" == *"qwen3_vl_2b_lgvq_single_metric_o2_16frame_54"* ]]; then
        state="RUNNING"
      else
        state="EXITED (PID RECYCLED)"
      fi
    fi
    printf '\n[%s] %s PID=%s\n' "${name}" "${state}" "${pid:-invalid}"
    if [[ -f "${RUN_DIR}/${name}.log" ]]; then
      tail -n "${TAIL_LINES}" "${RUN_DIR}/${name}.log" || true
    else
      printf 'log is not present yet\n'
    fi
  done

  printf '\nResult summaries already written:\n'
  find "${PROJECT_DIR}/runs" -maxdepth 2 -type f \
    \( -name 'training_summary.json' -o -name 'metrics_best_observed_test_optical_on.json' \
       -o -name 'optical_contribution_same_checkpoint.json' \) \
    -printf '  %TY-%Tm-%Td %TH:%TM  %p\n' 2>/dev/null | sort || true
}

while true; do
  snapshot
  (( INTERVAL > 0 )) || break
  sleep "${INTERVAL}"
done
