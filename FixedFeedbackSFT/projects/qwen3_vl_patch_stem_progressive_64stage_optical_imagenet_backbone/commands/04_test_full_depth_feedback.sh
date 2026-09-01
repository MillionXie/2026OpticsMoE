#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

"${PYTHON_BIN}" -m pytest -q \
  "${EXPERIMENT}/tests/test_full_depth_feedback.py" \
  "${EXPERIMENT}/tests/test_progressive_model.py"
