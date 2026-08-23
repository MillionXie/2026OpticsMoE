#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
ensure_stem

PHYSICAL_GPU_INDICES="${PHYSICAL_GPU_INDICES:?Set two free physical GPU indices, for example 3,5}"
IFS=',' read -r -a indices <<< "${PHYSICAL_GPU_INDICES}"
if [[ "${#indices[@]}" -ne 2 ]]; then
  echo "The controlled run requires exactly two GPUs and batch 96 per rank." >&2
  exit 1
fi
uuids=()
for index in "${indices[@]}"; do
  uuids+=("$(gpu_uuid "${index}")")
done
export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${uuids[*]}")"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

CONFIG="${EXPERIMENT}/configs/pretrain_90e_bs96.yaml"
RUN_DIR="${EXPERIMENT}/runs/p10_imagenet1k_pretrain_bs96_90e"
LOG="${EXPERIMENT}/logs/p10_imagenet1k_pretrain_bs96_90e.log"
PID_FILE="${RUN_DIR}/launch.pid"
TORCHRUN_BIN="$(dirname "${PYTHON_BIN}")/torchrun"
mkdir -p "${RUN_DIR}" "$(dirname "${LOG}")"
if [[ -f "${PID_FILE}" ]]; then
  previous_pid="$(cat "${PID_FILE}")"
  if [[ -n "${previous_pid}" ]] && kill -0 "${previous_pid}" 2>/dev/null; then
    echo "P10 is already running as PID ${previous_pid}."
    exit 1
  fi
fi
nohup "${TORCHRUN_BIN}" --standalone --nproc_per_node=2 \
  -m "${EXPERIMENT//\//.}.train" --config "${CONFIG}" --resume \
  > "${LOG}" 2>&1 < /dev/null &
launch_pid=$!
printf '%s\n' "${launch_pid}" > "${PID_FILE}"
sleep 8
kill -0 "${launch_pid}"
echo "Started P10 PID=${launch_pid}, GPUs=${PHYSICAL_GPU_INDICES}, global_batch=192"
echo "LOG=${LOG}"
