#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=SERVER_RFL_ENV.sh
source "${HERE}/SERVER_RFL_ENV.sh"

PORT="${1:-24515}"
GRAPHICS_ADAPTER="${CARLA_GRAPHICS_ADAPTER:-0}"
LOG="${CARLA_LOG:-${CARLA_ROOT}/carla_${PORT}.log}"

nohup "${CARLA_SERVER}" \
  -RenderOffScreen \
  -nosound \
  -quality-level=Low \
  "-carla-port=${PORT}" \
  "-graphicsadapter=${GRAPHICS_ADAPTER}" \
  >"${LOG}" 2>&1 &

echo "CARLA PID=$! port=${PORT} graphicsadapter=${GRAPHICS_ADAPTER}"
echo "CARLA log=${LOG}"
echo "Initial map loading can take several minutes on network storage."
