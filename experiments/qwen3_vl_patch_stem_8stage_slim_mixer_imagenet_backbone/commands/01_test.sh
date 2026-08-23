#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
ensure_stem
"${PYTHON_BIN}" -m pytest -q "${EXPERIMENT}/tests" \
  "${P08_EXPERIMENT}/tests"
