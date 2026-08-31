#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_training_common.sh"
require_training_sources

P11_CONTROL_GPU="${P11_CONTROL_GPU:?Set one free physical GPU index}"
P11_ACTION="${P11_ACTION:?Set P11_ACTION=fresh or P11_ACTION=resume explicitly}"
CONFIG="${EXPERIMENT}/configs/p11_epoch88_matched_continue_20e_gb192.yaml"
RUN_DIR="${EXPERIMENT}/runs/p11_epoch88_matched_continue_20e_gb192"
BASE_LOG="${EXPERIMENT}/logs/p11_epoch88_matched_continue_20e_gb192.log"
LATEST_LOG="${EXPERIMENT}/logs/p11_epoch88_matched_continue_20e_gb192.latest.log"
PID_FILE="${RUN_DIR}/launch.pid"
LOCK_FILE="${RUN_DIR}/launch.lock"
acquire_launch_lock "${LOCK_FILE}"
MODE="$(training_mode_argument "${P11_ACTION}" "${RUN_DIR}")"
export CUDA_VISIBLE_DEVICES="$(visible_gpu_uuids "${P11_CONTROL_GPU}")"
mkdir -p "${RUN_DIR}" "$(dirname "${BASE_LOG}")"
LOG="$(segmented_log_path "${BASE_LOG}" "${P11_ACTION}")"
ln -sfn "$(basename "${LOG}")" "${LATEST_LOG}"
printf '[launch] utc=%s action=%s GPU=%s config=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${P11_ACTION}" \
  "${P11_CONTROL_GPU}" "${CONFIG}" > "${LOG}"

nohup "${PYTHON_BIN}" -m "${EXPERIMENT//\//.}.p11_matched_continue" \
  --config "${CONFIG}" "${MODE}" \
  >> "${LOG}" 2>&1 < /dev/null 9>&9 &
launch_pid=$!
pid_tmp="${PID_FILE}.$$"
printf '%s\n' "${launch_pid}" > "${pid_tmp}"
mv "${pid_tmp}" "${PID_FILE}"
sleep 8
kill -0 "${launch_pid}"
echo "Started matched P11 PID=${launch_pid}, GPU=${P11_CONTROL_GPU}, effective_global_batch=192"
echo "LOG=${LOG}"
