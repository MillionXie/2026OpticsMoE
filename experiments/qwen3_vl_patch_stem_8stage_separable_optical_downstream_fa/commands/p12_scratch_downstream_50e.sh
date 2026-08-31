#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="${P12_SCRATCH_REPO_ROOT:-$(cd "${EXPERIMENT_DIR}/../.." && pwd)}"
PYTHON_BIN="${P12_SCRATCH_PYTHON_BIN:-/home/guest3/miniconda3/envs/xml/bin/python}"
BASE_CONFIG="${P12_SCRATCH_BASE_CONFIG:-${EXPERIMENT_DIR}/configs/base_50e.yaml}"
INIT_SEED="${P12_SCRATCH_INIT_SEED:-2026}"
RUN_ROOT="${P12_SCRATCH_RUN_ROOT:-${EXPERIMENT_DIR}/runs/p12_scratch_p11_body_seed_${INIT_SEED}_50e}"
SOURCE_CHECKPOINT="${P12_SCRATCH_SOURCE:-${EXPERIMENT_DIR}/runs/p12_scratch_sources/p11_body_seed_${INIT_SEED}/backbone.pt}"
CONFIG="${P12_SCRATCH_CONFIG:-${RUN_ROOT}/provenance/resolved_config.yaml}"
GPU_LIST="${P12_SCRATCH_GPU_LIST:-0,1,3,4,5}"
SEEDS="${P12_SCRATCH_SEEDS:-2026}"
ADAPTATION_SEEDS="${P12_SCRATCH_ADAPTATION_SEEDS:-${SEEDS}}"
POLL_SECONDS="${P12_SCRATCH_POLL_SECONDS:-20}"
MAX_RETRIES="${P12_SCRATCH_MAX_RETRIES:-2}"
MODULE="experiments.qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa"
QUEUE_DIR="${RUN_ROOT}/queue"
QUEUE_LOG="${QUEUE_DIR}/launcher.log"
QUEUE_PID_FILE="${QUEUE_DIR}/launcher.pid"

usage() {
  cat <<'EOF'
Usage:
  bash .../commands/p12_scratch_downstream_50e.sh prepare
  bash .../commands/p12_scratch_downstream_50e.sh launch
  bash .../commands/p12_scratch_downstream_50e.sh foreground
  bash .../commands/p12_scratch_downstream_50e.sh status
  bash .../commands/p12_scratch_downstream_50e.sh tail
  bash .../commands/p12_scratch_downstream_50e.sh summarize

Default matrix: 3 tasks x seed 2026 x 4 P12 groups. To expand the exact same
source to three downstream split seeds, set both seed variables to 2026,2027,2028:

  P12_SCRATCH_SEEDS=2026,2027,2028 \
  P12_SCRATCH_ADAPTATION_SEEDS=2026,2027,2028 \
  bash .../commands/p12_scratch_downstream_50e.sh launch

The frozen Qwen stem is retained. No P11 ImageNet-backbone checkpoint is read.
Within this control only, the existing `fa_pretrained` method key means fixed
feedback from the fresh source initialization (`FA-source-init`).
EOF
}

require_python() {
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
  fi
}

require_prepared() {
  require_python
  if [[ ! -f "${SOURCE_CHECKPOINT}" ]]; then
    echo "Scratch source is missing; run the prepare action first: ${SOURCE_CHECKPOINT}" >&2
    exit 1
  fi
  if [[ ! -f "${CONFIG}" ]]; then
    echo "Resolved scratch config is missing; run prepare first: ${CONFIG}" >&2
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
  prepare)
    require_python
    cd "${REPO_ROOT}"
    "${PYTHON_BIN}" -m "${MODULE}.scratch_source" export \
      --base-config "${BASE_CONFIG}" \
      --init-seed "${INIT_SEED}" \
      --output "${SOURCE_CHECKPOINT}"
    "${PYTHON_BIN}" -m "${MODULE}.scratch_source" render-config \
      --base-config "${BASE_CONFIG}" \
      --source-checkpoint "${SOURCE_CHECKPOINT}" \
      --output "${CONFIG}" \
      --output-root "${RUN_ROOT}"
    echo "Prepared a SHA-locked scratch source/config without an ImageNet body checkpoint."
    ;;
  launch)
    require_prepared
    mkdir -p "${QUEUE_DIR}"
    if [[ -f "${QUEUE_PID_FILE}" ]]; then
      previous_pid="$(tr -dc '0-9' < "${QUEUE_PID_FILE}")"
      if [[ -n "${previous_pid}" ]] && kill -0 "${previous_pid}" 2>/dev/null; then
        echo "P12 scratch queue is already running as PID ${previous_pid}." >&2
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
        echo "Scratch queue exited during startup; inspect ${QUEUE_LOG}." >&2
        tail -n 80 "${QUEUE_LOG}" >&2 || true
        exit 1
      fi
      sleep 1
    done
    echo "Started scratch queue PID=${queue_pid}; GPUs=${GPU_LIST}; seeds=${SEEDS}."
    echo "Queue log: ${QUEUE_LOG}"
    ;;
  foreground)
    require_prepared
    cd "${REPO_ROOT}"
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export PYTHONUNBUFFERED=1
    exec "${PYTHON_BIN}" -m "${MODULE}.queue" "${queue_args[@]}"
    ;;
  status)
    require_prepared
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
    require_prepared
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
