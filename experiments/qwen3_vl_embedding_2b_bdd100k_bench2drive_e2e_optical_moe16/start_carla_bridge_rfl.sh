#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
# shellcheck source=SERVER_RFL_ENV.sh
source "${HERE}/SERVER_RFL_ENV.sh"

CARLA_PORT="${CARLA_PORT:-24515}"
BRIDGE_PORT="${CARLA_BRIDGE_PORT:-24615}"
BRIDGE_HOST="${CARLA_BRIDGE_HOST:-127.0.0.1}"
AUTHKEY="${CARLA_BRIDGE_AUTHKEY:-bench2drive-local}"
MAP="${CARLA_MAP:-Town10HD_Opt}"
RFL_PYTHON="${RFL_PYTHON:-/home/guest3/miniconda3/envs/RFL/bin/python}"
RUNTIME_DIR="${CARLA_RUNTIME_DIR:-${HERE}/runs/carla_runtime}"
mkdir -p "${RUNTIME_DIR}"

export CARLA_GRAPHICS_ADAPTER="${CARLA_GRAPHICS_ADAPTER:-1}"
export CARLA_LOG="${RUNTIME_DIR}/carla_${CARLA_PORT}.log"
EXISTING_CARLA_PID="$(pgrep -f "CarlaUE4-Linux-Shipping.*-carla-port=${CARLA_PORT}" | tail -1 || true)"
if [[ -n "${EXISTING_CARLA_PID}" ]]; then
  echo "Reusing CARLA PID=${EXISTING_CARLA_PID} on port=${CARLA_PORT}"
else
  "${HERE}/start_carla_rfl.sh" "${CARLA_PORT}" | tee "${RUNTIME_DIR}/carla_start.txt"
fi

echo "Waiting for CARLA ${CARLA_PORT}..."
for attempt in $(seq 1 60); do
  if "${RFL_PYTHON}" "${HERE}/carla_runtime_check.py" \
      --host 127.0.0.1 --port "${CARLA_PORT}" --timeout 5 \
      >"${RUNTIME_DIR}/carla_check.log" 2>&1; then
    break
  fi
  if [[ "${attempt}" == "60" ]]; then
    echo "CARLA failed to become ready; see ${CARLA_LOG}" >&2
    exit 1
  fi
  sleep 2
done

# CarlaUE4.sh forks the actual shipping binary, so $! is only the short-lived
# launcher. Persist the real process that owns the RPC port for reliable stop.
CARLA_PID="$(pgrep -f "CarlaUE4-Linux-Shipping.*-carla-port=${CARLA_PORT}" | tail -1)"
if [[ -z "${CARLA_PID}" ]]; then
  echo "CARLA runtime check passed but the server PID could not be located" >&2
  exit 1
fi
echo "${CARLA_PID}" > "${RUNTIME_DIR}/carla.pid"
echo "${CARLA_PORT}" > "${RUNTIME_DIR}/carla.port"

cd "${REPO_ROOT}"
# Launch by file path: importing the Python-3.11 experiment package from the
# Python-3.8 CARLA environment would parse unrelated modern type annotations.
nohup "${RFL_PYTHON}" "${HERE}/carla_env_server.py" \
  --carla-host 127.0.0.1 \
  --carla-port "${CARLA_PORT}" \
  --bridge-host "${BRIDGE_HOST}" \
  --bridge-port "${BRIDGE_PORT}" \
  --authkey "${AUTHKEY}" \
  --map "${MAP}" \
  >"${RUNTIME_DIR}/carla_bridge.log" 2>&1 &
BRIDGE_PID=$!
echo "${BRIDGE_PID}" > "${RUNTIME_DIR}/bridge.pid"

sleep 3
if ! kill -0 "${BRIDGE_PID}" 2>/dev/null; then
  echo "CARLA bridge exited during startup:" >&2
  tail -100 "${RUNTIME_DIR}/carla_bridge.log" >&2
  exit 1
fi

echo "CARLA service PID=${CARLA_PID} port=${CARLA_PORT} adapter=${CARLA_GRAPHICS_ADAPTER}"
echo "Bridge service PID=${BRIDGE_PID} address=${BRIDGE_HOST}:${BRIDGE_PORT}"
echo "Logs: ${RUNTIME_DIR}"
