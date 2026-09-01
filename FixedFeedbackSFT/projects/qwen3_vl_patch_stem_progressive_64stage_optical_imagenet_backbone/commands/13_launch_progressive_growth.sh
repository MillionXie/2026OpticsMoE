#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_progressive_growth_common.sh"

TARGET_DEPTH="${TARGET_DEPTH:?Set TARGET_DEPTH=32, 64, or 100 explicitly}"
PHYSICAL_GPU_INDICES="${PHYSICAL_GPU_INDICES:?Set exactly four free physical GPU indices, e.g. 0,1,3,4}"
P13_ACTION="${P13_ACTION:?Set P13_ACTION=fresh or P13_ACTION=resume explicitly}"
select_progressive_growth_stage "${TARGET_DEPTH}"

IFS=',' read -r -a indices <<< "${PHYSICAL_GPU_INDICES}"
if [[ "${#indices[@]}" -ne 4 ]]; then
  echo "The controlled progressive run requires exactly four GPUs." >&2
  exit 1
fi

# This re-hashes and re-validates the fixed previous-depth best checkpoint on
# every launch. It creates a config only after that checkpoint exists, and an
# existing config must remain byte-semantically identical to the source identity.
acquire_launch_lock "${TARGET_LOCK_FILE}"
render_or_verify_progressive_config "${TARGET_DEPTH}"
MODE="$(training_mode_argument "${P13_ACTION}" "${TARGET_RUN_DIR}")"
export CUDA_VISIBLE_DEVICES="$(visible_gpu_uuids "${PHYSICAL_GPU_INDICES}")"
TORCHRUN_BIN="$(dirname "${PYTHON_BIN}")/torchrun"
mkdir -p "${TARGET_RUN_DIR}" "$(dirname "${TARGET_BASE_LOG}")"
PID_FILE="${TARGET_PID_FILE}"
LOG="$(segmented_log_path "${TARGET_BASE_LOG}" "${P13_ACTION}")"
ln -sfn "$(basename "${LOG}")" "${TARGET_LATEST_LOG}"
printf '[launch] utc=%s action=%s target_depth=%s GPUs=%s config=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${P13_ACTION}" "${TARGET_DEPTH}" \
  "${PHYSICAL_GPU_INDICES}" "${TARGET_CONFIG}" > "${LOG}"

nohup "${TORCHRUN_BIN}" --standalone --nproc_per_node=4 \
  -m "${MODULE}.train" --config "${TARGET_CONFIG}" "${MODE}" \
  >> "${LOG}" 2>&1 < /dev/null 9>&9 &
launch_pid=$!
pid_tmp="${PID_FILE}.$$"
printf '%s\n' "${launch_pid}" > "${pid_tmp}"
mv "${pid_tmp}" "${PID_FILE}"
sleep 8
kill -0 "${launch_pid}"
echo "Started guarded P13 ${TARGET_DEPTH}-stage growth PID=${launch_pid}, GPUs=${PHYSICAL_GPU_INDICES}, effective_global_batch=192"
echo "PARENT=${PARENT_CHECKPOINT}"
echo "CONFIG=${TARGET_CONFIG}"
echo "LOG=${LOG}"
