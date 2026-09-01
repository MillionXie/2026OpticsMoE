#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="${P12_PHASE_ONLY_REPO_ROOT:-$(cd "${EXPERIMENT_DIR}/../../.." && pwd)}"
RUNS_DIR="${P12_PHASE_ONLY_RUNS_ROOT:-${REPO_ROOT}/FixedFeedbackSFT/runs/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa}"
PYTHON_BIN="${P12_PHASE_ONLY_PYTHON_BIN:-/home/guest3/miniconda3/envs/xml/bin/python}"
CONFIG="${P12_PHASE_ONLY_CONFIG:-${EXPERIMENT_DIR}/configs/phase_only_50e.yaml}"
GPU_LIST="${P12_PHASE_ONLY_GPU_LIST:-0,1,3,4,5}"
SEEDS="${P12_PHASE_ONLY_SEEDS:-2026,2027,2028}"
ADAPTATION_SEEDS="${P12_PHASE_ONLY_ADAPTATION_SEEDS:-2026,2027,2028}"
POLL_SECONDS="${P12_PHASE_ONLY_POLL_SECONDS:-20}"
MAX_RETRIES="${P12_PHASE_ONLY_MAX_RETRIES:-2}"
TASK="${P12_PHASE_ONLY_TASK:-caltech101}"
METHOD="${P12_PHASE_ONLY_METHOD:-bp}"
SEED="${P12_PHASE_ONLY_SEED:-2026}"
MODULE="experiments.qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa"
RUN_ROOT="${RUNS_DIR}/p12_phase_only_fa_50e"
QUEUE_DIR="${RUN_ROOT}/queue"
QUEUE_LOG="${QUEUE_DIR}/launcher.log"
QUEUE_PID_FILE="${QUEUE_DIR}/launcher.pid"

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

require_assets() {
  [[ -x "${PYTHON_BIN}" ]] || { echo "Python not found: ${PYTHON_BIN}" >&2; exit 1; }
  [[ -f "${CONFIG}" ]] || { echo "Config not found: ${CONFIG}" >&2; exit 1; }
  command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi is required" >&2; exit 1; }
}

action="${1:-status}"
case "${action}" in
  launch)
    require_assets
    mkdir -p "${QUEUE_DIR}"
    if [[ -f "${QUEUE_PID_FILE}" ]]; then
      previous_pid="$(tr -dc '0-9' < "${QUEUE_PID_FILE}")"
      if [[ -n "${previous_pid}" ]] && kill -0 "${previous_pid}" 2>/dev/null; then
        echo "Phase-only queue already runs as PID ${previous_pid}." >&2
        exit 1
      fi
    fi
    nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv,noheader
    cd "${REPO_ROOT}"
    nohup env CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1 \
      "${PYTHON_BIN}" -m "${MODULE}.phase_only_queue" "${queue_args[@]}" \
      > "${QUEUE_LOG}" 2>&1 < /dev/null &
    queue_pid=$!
    printf '%s\n' "${queue_pid}" > "${QUEUE_PID_FILE}"
    sleep 5
    kill -0 "${queue_pid}" 2>/dev/null || {
      tail -n 80 "${QUEUE_LOG}" >&2 || true
      exit 1
    }
    echo "Started phase-only queue PID=${queue_pid}; GPUs=${GPU_LIST}."
    ;;
  foreground)
    require_assets
    cd "${REPO_ROOT}"
    export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
    exec "${PYTHON_BIN}" -m "${MODULE}.phase_only_queue" "${queue_args[@]}"
    ;;
  status)
    require_assets
    cd "${REPO_ROOT}"
    "${PYTHON_BIN}" -m "${MODULE}.phase_only_queue" "${queue_args[@]}" --status
    ;;
  tail)
    [[ -f "${QUEUE_PID_FILE}" ]] && echo "Wrapper PID: $(cat "${QUEUE_PID_FILE}")"
    nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv,noheader
    tail -n 80 "${QUEUE_LOG}" 2>/dev/null || true
    ;;
  run-one)
    require_assets
    cd "${REPO_ROOT}"
    exec "${PYTHON_BIN}" -m "${MODULE}.phase_only" \
      --config "${CONFIG}" --task "${TASK}" --method "${METHOD}" --seed "${SEED}" \
      --adaptation-scope phase_and_head --resume
    ;;
  smoke)
    require_assets
    cd "${REPO_ROOT}"
    exec "${PYTHON_BIN}" -m "${MODULE}.phase_only_smoke" \
      --config "${CONFIG}" --task "${TASK}" --method "${METHOD}" --seed "${SEED}" \
      --adaptation-scope phase_and_head \
      --output-root "${RUN_ROOT}_smoke" \
      --max-train-batches 1 --max-validation-batches 1 --max-test-batches 1
    ;;
  summarize)
    require_assets
    cd "${REPO_ROOT}"
    "${PYTHON_BIN}" -m "${MODULE}.phase_only_summarize" \
      --config "${CONFIG}" --seeds "${SEEDS}" \
      --adaptation-seeds "${ADAPTATION_SEEDS}" \
      --output "${RUN_ROOT}/summary.json"
    ;;
  help|-h|--help)
    echo "Usage: bash p12_phase_only_fa_50e.sh {launch|foreground|status|tail|run-one|smoke|summarize}"
    ;;
  *)
    echo "Unknown action: ${action}" >&2
    exit 2
    ;;
esac
