#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-30}"

while true; do
  clear || true
  date --iso-8601=seconds
  bash "${SCRIPT_DIR}/10_status_p11_matched.sh"
  sleep "${INTERVAL_SECONDS}"
done
