#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
ensure_frozen_p11_asset

P11_CLEAN_GPU="${P11_CLEAN_GPU:-5}"
P11_CLEAN_ACTION="${P11_CLEAN_ACTION:?Set fresh or resume explicitly}"
case "${P11_CLEAN_ACTION}" in
  fresh) MODE="--fresh" ;;
  resume) MODE="--resume" ;;
  *) echo "P11_CLEAN_ACTION must be fresh or resume." >&2; exit 1 ;;
esac
require_idle_gpu "${P11_CLEAN_GPU}"

CONFIG="${EXPERIMENT}/configs/clean_recovery_5e_raw_gpu5_gb96.yaml"
RUN="${RUNS_DIR}/p11_clean_recovery_5e_raw_gpu5_gb96"
SOURCE="${RUNS_DIR}/p11_large_recipe_proxy_5e_phase7e3_2gpu_gb384/checkpoints/last.pt"
EXPECTED_SOURCE_SHA256="34175ba9e764b7eef5bd59b1e1d1dd7f602281d02bd709ebf12ec55c0338f681"
LOG_DIR="${RUNS_DIR}/logs"
[[ -f "${SOURCE}" ]] || { echo "Missing source checkpoint: ${SOURCE}" >&2; exit 1; }
ACTUAL_SOURCE_SHA256="$(sha256sum "${SOURCE}" | awk '{print $1}')"
[[ "${ACTUAL_SOURCE_SHA256}" == "${EXPECTED_SOURCE_SHA256}" ]] || {
  echo "Source SHA256 mismatch: ${ACTUAL_SOURCE_SHA256}" >&2
  exit 1
}
if [[ "${P11_CLEAN_ACTION}" == "fresh" ]] && [[ -d "${RUN}" ]]; then
  FIRST_EXISTING="$(find "${RUN}" -mindepth 1 -maxdepth 1 -print -quit)"
  if [[ -n "${FIRST_EXISTING}" ]]; then
    echo "Fresh mode refuses the non-empty target directory ${RUN}." >&2
    echo "First existing entry: ${FIRST_EXISTING}" >&2
    exit 1
  fi
fi
if [[ "${P11_CLEAN_ACTION}" == "resume" ]] && \
  [[ ! -f "${RUN}/checkpoints/last.pt" ]]; then
  echo "Resume requires ${RUN}/checkpoints/last.pt." >&2
  exit 1
fi
if [[ -f "${RUN}/launch.pid" ]]; then
  OLD_PID="$(cat "${RUN}/launch.pid")"
  if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "Clean-recovery launcher is already live: PID=${OLD_PID}." >&2
    exit 1
  fi
fi
if command -v pgrep >/dev/null 2>&1; then
  EXISTING_PROCESS="$(pgrep -af "${MODULE}[.]clean_recovery" || true)"
  if [[ -n "${EXISTING_PROCESS}" ]]; then
    echo "A clean-recovery process already exists; refusing a duplicate launch:" >&2
    echo "${EXISTING_PROCESS}" >&2
    exit 1
  fi
fi

VISIBLE_UUID="$(gpu_uuid "${P11_CLEAN_GPU}")"
mkdir -p "${RUN}" "${LOG_DIR}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="${LOG_DIR}/p11_clean_recovery.${STAMP}.${P11_CLEAN_ACTION}.log"
printf '[launch] utc=%s action=%s physical_gpu=%s config=%s source_sha256=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${P11_CLEAN_ACTION}" \
  "${P11_CLEAN_GPU}" "${CONFIG}" "${ACTUAL_SOURCE_SHA256}" > "${LOG}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
unset WORLD_SIZE RANK LOCAL_RANK MASTER_ADDR MASTER_PORT
CUDA_VISIBLE_DEVICES="${VISIBLE_UUID}" nohup "${PYTHON_BIN}" -m \
  "${MODULE}.clean_recovery" --config "${CONFIG}" "${MODE}" \
  >> "${LOG}" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "${PID}" > "${RUN}/launch.pid.tmp"
mv "${RUN}/launch.pid.tmp" "${RUN}/launch.pid"
sleep 10
kill -0 "${PID}"
ln -sfn "$(basename "${LOG}")" "${LOG_DIR}/p11_clean_recovery.latest.log"
echo "Started P11 clean recovery PID=${PID}, physical GPU=${P11_CLEAN_GPU}, GB=96."
echo "Log: ${LOG}"
