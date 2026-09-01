#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
ensure_stem
ensure_frozen_p11_asset

P11_FORMAL_GPUS="${P11_FORMAL_GPUS:?Set exactly five comma-separated physical GPU indices}"
P11_FORMAL_ACTION="${P11_FORMAL_ACTION:?Set fresh or resume explicitly}"
P11_FORMAL_PHASE_LR="${P11_FORMAL_PHASE_LR:?Set 2e3 or 7e3 from the proxy decision}"
case "${P11_FORMAL_ACTION}" in
  fresh) MODE="--fresh" ;;
  resume) MODE="--resume" ;;
  *) echo "P11_FORMAL_ACTION must be fresh or resume." >&2; exit 1 ;;
esac
IFS=',' read -r -a indices <<< "${P11_FORMAL_GPUS}"
[[ "${#indices[@]}" -eq 5 ]] || { echo "Formal run requires exactly five GPUs." >&2; exit 1; }
declare -A seen=()
uuids=()
for index in "${indices[@]}"; do
  [[ "${index}" =~ ^[0-9]+$ ]] || { echo "Invalid GPU index ${index}." >&2; exit 1; }
  [[ -z "${seen[${index}]+x}" ]] || { echo "Duplicate GPU index ${index}." >&2; exit 1; }
  seen[${index}]=1
  require_idle_gpu "${index}"
  uuids+=("$(gpu_uuid "${index}")")
done
VISIBLE="$(IFS=','; echo "${uuids[*]}")"

case "${P11_FORMAL_PHASE_LR}" in
  2e3)
    CONFIG="${EXPERIMENT}/configs/large_recipe_formal_100e_phase2e3_5gpu_gb480.yaml"
    RUN_NAME="p11_large_recipe_formal_100e_phase2e3_5gpu_gb480"
    ;;
  7e3)
    CONFIG="${EXPERIMENT}/configs/large_recipe_formal_100e_phase7e3_5gpu_gb480.yaml"
    RUN_NAME="p11_large_recipe_formal_100e_phase7e3_5gpu_gb480"
    ;;
  *) echo "P11_FORMAL_PHASE_LR must be exactly 2e3 or 7e3." >&2; exit 1 ;;
esac
RUN_DIR="${RUNS_DIR}/${RUN_NAME}"
LOG_DIR="${RUNS_DIR}/logs"
mkdir -p "${RUN_DIR}" "${LOG_DIR}"
if [[ "${P11_FORMAL_ACTION}" == "fresh" ]] && \
  [[ -e "${RUN_DIR}/manifest.json" || -d "${RUN_DIR}/checkpoints" ]]; then
  echo "Fresh mode refuses existing run artifacts in ${RUN_DIR}." >&2
  exit 1
fi
if [[ "${P11_FORMAL_ACTION}" == "resume" ]] && \
  [[ ! -f "${RUN_DIR}/checkpoints/last.pt" ]]; then
  echo "Resume requires ${RUN_DIR}/checkpoints/last.pt." >&2
  exit 1
fi
if [[ -f "${RUN_DIR}/launch.pid" ]]; then
  old_pid="$(cat "${RUN_DIR}/launch.pid")"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "Formal run already has live launcher PID ${old_pid}." >&2
    exit 1
  fi
fi
TORCHRUN_BIN="$(dirname "${PYTHON_BIN}")/torchrun"
[[ -x "${TORCHRUN_BIN}" ]] || { echo "Missing torchrun: ${TORCHRUN_BIN}" >&2; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="${LOG_DIR}/p11_formal_100e_phase${P11_FORMAL_PHASE_LR}_5gpu.${STAMP}.${P11_FORMAL_ACTION}.log"
printf '[launch] utc=%s action=%s physical_gpus=%s config=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${P11_FORMAL_ACTION}" \
  "${P11_FORMAL_GPUS}" "${CONFIG}" > "${LOG}"
CUDA_VISIBLE_DEVICES="${VISIBLE}" nohup "${TORCHRUN_BIN}" --standalone \
  --nproc_per_node=5 -m "${MODULE}.large_scale_continue" \
  --config "${CONFIG}" "${MODE}" >> "${LOG}" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "${PID}" > "${RUN_DIR}/launch.pid.tmp"
mv "${RUN_DIR}/launch.pid.tmp" "${RUN_DIR}/launch.pid"
sleep 10
kill -0 "${PID}"
ln -sfn "$(basename "${LOG}")" "${LOG_DIR}/p11_formal_100e_phase${P11_FORMAL_PHASE_LR}.latest.log"
echo "Started formal P11 recipe PID=${PID}, phase=${P11_FORMAL_PHASE_LR}, GPUs=${P11_FORMAL_GPUS}, GB=480."
echo "LOG=${LOG}"
