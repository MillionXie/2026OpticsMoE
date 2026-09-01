#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="${P12_REPO_ROOT:-$(cd "${EXPERIMENT_DIR}/../../.." && pwd)}"
RUNS_DIR="${P12_RUNS_ROOT:-${REPO_ROOT}/FixedFeedbackSFT/runs/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa}"
PYTHON_BIN="${P12_PYTHON_BIN:-/home/guest3/miniconda3/envs/xml/bin/python}"
CONFIG="${P12_CONFIG:-${EXPERIMENT_DIR}/configs/base_50e.yaml}"
GPU_LIST="${P12_GPU_LIST:-1,2,3,4,5}"
SEEDS="${P12_SEEDS:-2026,2027,2028}"
ADAPTATION_SEEDS="${P12_ADAPTATION_SEEDS:-2026}"
POLL_SECONDS="${P12_POLL_SECONDS:-20}"
MAX_RETRIES="${P12_MAX_RETRIES:-2}"
MODULE="experiments.qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa"
RUN_ROOT="${RUNS_DIR}/p12_downstream_fa_50e"
QUEUE_DIR="${RUN_ROOT}/queue"
QUEUE_LOG="${QUEUE_DIR}/launcher.log"
QUEUE_PID_FILE="${QUEUE_DIR}/launcher.pid"

usage() {
  cat <<'EOF'
Usage:
  bash .../commands/p12_downstream_fa_50e.sh launch
  bash .../commands/p12_downstream_fa_50e.sh foreground
  bash .../commands/p12_downstream_fa_50e.sh status
  bash .../commands/p12_downstream_fa_50e.sh tail
  bash .../commands/p12_downstream_fa_50e.sh summarize

Environment overrides:
  P12_GPU_LIST=1,2,3,4,5
  P12_SEEDS=2026,2027,2028
  P12_ADAPTATION_SEEDS=2026
  P12_PYTHON_BIN=/home/guest3/miniconda3/envs/xml/bin/python
  P12_REPO_ROOT=/DATA/DATA1/guest3/2026OpticsMoE
  P12_POLL_SECONDS=20
  P12_MAX_RETRIES=2

The default first formal wave creates all nine NoFT/common starts but adapts
only seed 2026. After the preregistered pilot gate, set
P12_ADAPTATION_SEEDS=2026,2027,2028 to expand the same recipe.
EOF
}

require_assets() {
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
  fi
  if [[ ! -f "${CONFIG}" ]]; then
    echo "P12 config not found: ${CONFIG}" >&2
    exit 1
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required for safe GPU ownership checks." >&2
    exit 1
  fi
}

queue_args=(
  --config "${CONFIG}"
  --gpus "${GPU_LIST}"
  --seeds "${SEEDS}"
  --adaptation-seeds "${ADAPTATION_SEEDS}"
  --python "${PYTHON_BIN}"
  --repo-root "${REPO_ROOT}"
  --poll-seconds "${POLL_SECONDS}"
  --max-retries "${MAX_RETRIES}"
)

action="${1:-status}"
case "${action}" in
  launch)
    require_assets
    mkdir -p "${QUEUE_DIR}"
    if [[ -f "${QUEUE_PID_FILE}" ]]; then
      previous_pid="$(tr -dc '0-9' < "${QUEUE_PID_FILE}")"
      if [[ -n "${previous_pid}" ]] && kill -0 "${previous_pid}" 2>/dev/null; then
        echo "P12 queue wrapper is already running as PID ${previous_pid}." >&2
        exit 1
      fi
    fi
    echo "Pre-launch GPU snapshot (queue.py rechecks compute-app ownership):"
    nvidia-smi \
      --query-gpu=index,uuid,name,memory.used,utilization.gpu \
      --format=csv,noheader
    cd "${REPO_ROOT}"
    nohup env CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1 \
      "${PYTHON_BIN}" -m "${MODULE}.queue" "${queue_args[@]}" \
      > "${QUEUE_LOG}" 2>&1 < /dev/null &
    queue_pid=$!
    printf '%s\n' "${queue_pid}" > "${QUEUE_PID_FILE}"
    for _ in 1 2 3 4 5; do
      if ! kill -0 "${queue_pid}" 2>/dev/null; then
        echo "P12 queue exited during startup; inspect ${QUEUE_LOG}." >&2
        tail -n 80 "${QUEUE_LOG}" >&2 || true
        exit 1
      fi
      sleep 1
    done
    echo "Started P12 queue PID=${queue_pid}; GPUs=${GPU_LIST}; common seeds=${SEEDS}; adaptation seeds=${ADAPTATION_SEEDS}."
    echo "Queue log: ${QUEUE_LOG}"
    ;;
  foreground)
    require_assets
    cd "${REPO_ROOT}"
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export PYTHONUNBUFFERED=1
    exec "${PYTHON_BIN}" -m "${MODULE}.queue" "${queue_args[@]}"
    ;;
  status)
    require_assets
    cd "${REPO_ROOT}"
    "${PYTHON_BIN}" -m "${MODULE}.queue" "${queue_args[@]}" --status
    ;;
  tail)
    if [[ -f "${QUEUE_PID_FILE}" ]]; then
      echo "Wrapper PID: $(cat "${QUEUE_PID_FILE}")"
    else
      echo "Wrapper PID: not launched by this script"
    fi
    nvidia-smi \
      --query-gpu=index,uuid,name,memory.used,utilization.gpu \
      --format=csv,noheader
    tail -n 80 "${QUEUE_LOG}" 2>/dev/null || true
    ;;
  summarize)
    require_assets
    cd "${REPO_ROOT}"
    "${PYTHON_BIN}" -m "${MODULE}.summarize" \
      --config "${CONFIG}" \
      --seeds "${SEEDS}" \
      --adaptation-seeds "${ADAPTATION_SEEDS}" \
      --output "${RUN_ROOT}/summary.json"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown action: ${action}" >&2
    usage >&2
    exit 2
    ;;
esac
