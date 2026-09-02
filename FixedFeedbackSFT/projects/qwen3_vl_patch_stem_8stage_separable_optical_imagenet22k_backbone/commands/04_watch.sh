#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

LOG="${IN22K_LOG:-}"
if [[ -z "${LOG}" ]]; then
  LOG="$(find "${LOG_DIR}" -maxdepth 1 -type f -name '*.log' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
fi
if [[ -z "${LOG}" || ! -f "${LOG}" ]]; then
  echo "No ImageNet-large training log found" >&2
  exit 1
fi
echo "Watching ${LOG}"
tail -n "${IN22K_WATCH_LINES:-120}" -f "${LOG}"
