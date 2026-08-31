#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_training_common.sh"
require_training_sources

PHYSICAL_GPU_INDICES="${PHYSICAL_GPU_INDICES:?Set exactly four free physical GPU indices, e.g. 0,1,3,4}"
P13_ACTION="${P13_ACTION:?Set P13_ACTION=fresh or P13_ACTION=resume explicitly}"
IFS=',' read -r -a indices <<< "${PHYSICAL_GPU_INDICES}"
if [[ "${#indices[@]}" -ne 4 ]]; then
  echo "The controlled P13 run requires exactly four GPUs." >&2
  exit 1
fi

CONFIG="${EXPERIMENT}/configs/growth16_fa_source_20e_gb192.yaml"
RUN_DIR="${EXPERIMENT}/runs/p13_growth16_fa_source_20e_gb192"
BASE_LOG="${EXPERIMENT}/logs/p13_growth16_fa_source_20e_gb192.log"
LATEST_LOG="${EXPERIMENT}/logs/p13_growth16_fa_source_20e_gb192.latest.log"
PID_FILE="${RUN_DIR}/launch.pid"
LOCK_FILE="${RUN_DIR}/launch.lock"
acquire_launch_lock "${LOCK_FILE}"
MODE="$(training_mode_argument "${P13_ACTION}" "${RUN_DIR}")"
export CUDA_VISIBLE_DEVICES="$(visible_gpu_uuids "${PHYSICAL_GPU_INDICES}")"
TORCHRUN_BIN="$(dirname "${PYTHON_BIN}")/torchrun"
mkdir -p "${RUN_DIR}" "$(dirname "${BASE_LOG}")"
LOG="$(segmented_log_path "${BASE_LOG}" "${P13_ACTION}")"
ln -sfn "$(basename "${LOG}")" "${LATEST_LOG}"
printf '[launch] utc=%s action=%s GPUs=%s config=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${P13_ACTION}" \
  "${PHYSICAL_GPU_INDICES}" "${CONFIG}" > "${LOG}"

nohup "${TORCHRUN_BIN}" --standalone --nproc_per_node=4 \
  -m "${EXPERIMENT//\//.}.train" --config "${CONFIG}" "${MODE}" \
  >> "${LOG}" 2>&1 < /dev/null 9>&9 &
launch_pid=$!
pid_tmp="${PID_FILE}.$$"
printf '%s\n' "${launch_pid}" > "${pid_tmp}"
mv "${pid_tmp}" "${PID_FILE}"
sleep 8
kill -0 "${launch_pid}"
echo "Started P13 growth PID=${launch_pid}, GPUs=${PHYSICAL_GPU_INDICES}, effective_global_batch=192"
echo "LOG=${LOG}"
