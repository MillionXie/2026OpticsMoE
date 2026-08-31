#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="${P12_PHASE_ONLY_REPO_ROOT:-$(cd "${EXPERIMENT_DIR}/../.." && pwd)}"
WRAPPER="${SCRIPT_DIR}/p12_phase_only_fa_50e.sh"
GPU_LIST="${P12_PHASE_ONLY_ISIC_RETRY_GPUS:-0,3,5}"
SEED="${P12_PHASE_ONLY_ISIC_RETRY_SEED:-2026}"
RUN_ROOT="${EXPERIMENT_DIR}/runs/p12_phase_only_fa_50e/isic2016"
METHODS=(bp fa_pretrained fa_random)

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [[ ${#GPUS[@]} -ne ${#METHODS[@]} ]]; then
  echo "Expected exactly three GPUs for ${METHODS[*]}; got ${GPU_LIST}" >&2
  exit 2
fi

status_one() {
  local method="$1"
  local run_dir="${RUN_ROOT}/${method}/seed_${SEED}"
  local pid_file="${run_dir}/retry_after_numeric_fix.pid"
  local result_file="${run_dir}/result.json"
  local pid=""
  [[ -f "${pid_file}" ]] && pid="$(tr -dc '0-9' < "${pid_file}")"
  if [[ -f "${result_file}" ]]; then
    echo "${method}: complete (${result_file})"
  elif [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    echo "${method}: running pid=${pid}"
  else
    echo "${method}: not running"
  fi
}

action="${1:-status}"
case "${action}" in
  launch)
    cd "${REPO_ROOT}"
    for index in "${!METHODS[@]}"; do
      method="${METHODS[${index}]}"
      gpu="${GPUS[${index}]}"
      run_dir="${RUN_ROOT}/${method}/seed_${SEED}"
      result_file="${run_dir}/result.json"
      pid_file="${run_dir}/retry_after_numeric_fix.pid"
      log_file="${run_dir}/logs/retry_after_numeric_fix.log"
      mkdir -p "${run_dir}/logs"
      if [[ -f "${result_file}" ]]; then
        echo "Refusing to overwrite completed ${method}: ${result_file}" >&2
        exit 1
      fi
      if [[ -f "${pid_file}" ]]; then
        old_pid="$(tr -dc '0-9' < "${pid_file}")"
        if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
          echo "${method} already runs as PID ${old_pid}" >&2
          exit 1
        fi
      fi
      nohup env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu}" \
        P12_PHASE_ONLY_TASK=isic2016 P12_PHASE_ONLY_METHOD="${method}" \
        P12_PHASE_ONLY_SEED="${SEED}" \
        bash "${WRAPPER}" run-one > "${log_file}" 2>&1 < /dev/null &
      pid=$!
      printf '%s\n' "${pid}" > "${pid_file}"
      sleep 2
      if ! kill -0 "${pid}" 2>/dev/null; then
        tail -n 80 "${log_file}" >&2 || true
        exit 1
      fi
      echo "Started ${method} on physical GPU ${gpu}, PID=${pid}."
    done
    ;;
  status)
    for method in "${METHODS[@]}"; do
      status_one "${method}"
    done
    ;;
  tail)
    for method in "${METHODS[@]}"; do
      echo "===== ${method} ====="
      tail -n 25 "${RUN_ROOT}/${method}/seed_${SEED}/logs/retry_after_numeric_fix.log" 2>/dev/null || true
    done
    ;;
  *)
    echo "Usage: bash $(basename "$0") {launch|status|tail}" >&2
    exit 2
    ;;
esac
