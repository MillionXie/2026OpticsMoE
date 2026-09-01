#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
ensure_stem

PHYSICAL_GPU_INDICES="${PHYSICAL_GPU_INDICES:-3,5}"
IFS=',' read -r -a indices <<< "${PHYSICAL_GPU_INDICES}"
uuids=()
for index in "${indices[@]}"; do
  uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | sed -n "$((index + 1))p")"
  : "${uuid:?Could not resolve GPU ${index}}"
  uuids+=("${uuid}")
done
export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${uuids[*]}")"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

CONFIG="${EXPERIMENT}/configs/pretrain_90e_bs96.yaml"
RUN_DIR="${RUNS_DIR}/p08_imagenet1k_pretrain_bs96_90e"
LOG="${RUNS_DIR}/logs/p08_imagenet1k_pretrain_bs96_90e.log"
PID_FILE="${RUN_DIR}/launch.pid"
TORCHRUN_BIN="$(dirname "${PYTHON_BIN}")/torchrun"
mkdir -p "${RUN_DIR}" "$(dirname "${LOG}")"

if [[ -f "${PID_FILE}" ]]; then
  previous_pid="$(cat "${PID_FILE}")"
  if [[ -n "${previous_pid}" ]] && kill -0 "${previous_pid}" 2>/dev/null; then
    echo "Training is already running as PID ${previous_pid}."
    exit 1
  fi
fi

nohup "${TORCHRUN_BIN}" --standalone --nproc_per_node="${#indices[@]}" \
  -m "${MODULE}.train" --config "${CONFIG}" --resume \
  > "${LOG}" 2>&1 < /dev/null &
launch_pid=$!
printf '%s\n' "${launch_pid}" > "${PID_FILE}"
sleep 5

if ! kill -0 "${launch_pid}" 2>/dev/null; then
  echo "Training exited during startup. Last log lines:"
  tail -n 80 "${LOG}"
  exit 1
fi

echo "Started optimized P08 pretraining."
echo "PID=${launch_pid} GPUs=${PHYSICAL_GPU_INDICES} per_rank_batch=96 global_batch=$((96 * ${#indices[@]}))"
echo "LOG=${LOG}"
