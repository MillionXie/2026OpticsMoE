#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

: "${METHOD:?Set METHOD to noft, bp, fa_pretrained, or fa_random}"
: "${RUN_SEEDS:?Set RUN_SEEDS to a comma-separated subset of 2026,2027,2028}"
: "${PHYSICAL_GPU_INDEX:?Set PHYSICAL_GPU_INDEX to the nvidia-smi GPU index}"

case "${METHOD}" in
  noft|bp|fa_pretrained|fa_random) ;;
  *) echo "Unsupported formal METHOD=${METHOD}" >&2; exit 2 ;;
esac

if [[ -n "${WAIT_PID:-}" ]]; then
  [[ "${WAIT_PID}" =~ ^[0-9]+$ ]] || {
    echo "WAIT_PID must be numeric, got ${WAIT_PID}" >&2
    exit 2
  }
  echo "Waiting for PID ${WAIT_PID} before starting ${METHOD}: ${RUN_SEEDS}"
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep 30
  done
fi

IFS=',' read -r -a seeds <<< "${RUN_SEEDS}"
for run_seed in "${seeds[@]}"; do
  case "${run_seed}" in
    2026|2027|2028) ;;
    *) echo "Unsupported formal seed=${run_seed}" >&2; exit 2 ;;
  esac
  echo "Starting queued formal run: method=${METHOD} seed=${run_seed} gpu=${PHYSICAL_GPU_INDEX}"
  METHOD="${METHOD}" RUN_SEED="${run_seed}" PHYSICAL_GPU_INDEX="${PHYSICAL_GPU_INDEX}" \
    bash "${SCRIPT_DIR}/29_run_a13_formal_method.sh"
done
