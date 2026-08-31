#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

"${PYTHON_BIN}" -m pytest -q "${EXPERIMENT}/tests"

