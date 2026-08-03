#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="${CARLA_RUNTIME_DIR:-${HERE}/runs/carla_runtime}"
for name in bridge carla; do
  path="${RUNTIME_DIR}/${name}.pid"
  if [[ -f "${path}" ]]; then
    pid="$(cat "${path}")"
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}"
      for _attempt in $(seq 1 10); do
        kill -0 "${pid}" 2>/dev/null || break
        sleep 0.5
      done
      if kill -0 "${pid}" 2>/dev/null; then
        kill -KILL "${pid}"
      fi
      echo "Stopped ${name} PID=${pid}"
    fi
    rm -f "${path}"
  fi
done

# Fall back to the exact configured CARLA RPC port when the shell launcher PID
# has already exited or an older runtime wrote the wrong wrapper PID.
port_path="${RUNTIME_DIR}/carla.port"
if [[ -f "${port_path}" ]]; then
  port="$(cat "${port_path}")"
  matching="$(pgrep -f "CarlaUE4-Linux-Shipping.*-carla-port=${port}" || true)"
  if [[ -n "${matching}" ]]; then
    kill ${matching} 2>/dev/null || true
    sleep 2
    matching="$(pgrep -f "CarlaUE4-Linux-Shipping.*-carla-port=${port}" || true)"
    [[ -z "${matching}" ]] || kill -KILL ${matching} 2>/dev/null || true
  fi
  rm -f "${port_path}"
fi
