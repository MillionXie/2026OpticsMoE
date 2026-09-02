#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

CONFIG="$(recipe_config)"
GPUS="${IN22K_GPUS:-0,1,2,3,5}"
ACTION="${IN22K_ACTION:-fresh}"

# Critical order: a missing/incorrect data manifest, source root, immutable
# asset, or disk budget fails here.  No output/log directory exists yet and no
# CUDA device has been selected or occupied.
preflight_cpu "${CONFIG}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
if (( ${#GPU_ARRAY[@]} != 5 )); then
  echo "The checked-in formal configs lock world_size=5; IN22K_GPUS needs 5 indices" >&2
  exit 2
fi
UUIDS=()
for gpu in "${GPU_ARRAY[@]}"; do
  require_idle_gpu "${gpu}"
  UUIDS+=("$(gpu_uuid "${gpu}")")
done
UUID_CSV="$(IFS=','; echo "${UUIDS[*]}")"

ARGS=(--config "${CONFIG}")
case "${ACTION}" in
  fresh) ;;
  resume) ARGS+=(--resume) ;;
  *) echo "IN22K_ACTION must be fresh or resume" >&2; exit 2 ;;
esac

mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/${LARGE_DATA_RECIPE:-fall11_full}_${STAMP}.log"
PID_FILE="${LOG}.pid"
nohup env CUDA_VISIBLE_DEVICES="${UUID_CSV}" \
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=5 \
  -m "${MODULE}" "${ARGS[@]}" >"${LOG}" 2>&1 &
PID="$!"
sleep 10
if ! kill -0 "${PID}" 2>/dev/null; then
  echo "Training process exited during the 10-second startup audit" >&2
  tail -n 120 "${LOG}" >&2 || true
  exit 1
fi
PID_TMP="${PID_FILE}.tmp-${PID}"
printf '%s\n' "${PID}" >"${PID_TMP}"
mv -f "${PID_TMP}" "${PID_FILE}"
LATEST_TMP="${LOG_DIR}/.latest.log.tmp-${PID}"
ln -s "$(basename "${LOG}")" "${LATEST_TMP}"
mv -Tf "${LATEST_TMP}" "${LOG_DIR}/latest.log"
echo "Started PID ${PID}"
echo "Log: ${LOG}"
echo "PID file: ${PID_FILE}"
