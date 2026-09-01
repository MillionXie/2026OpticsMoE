#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPOSITORY_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
"${PYTHON_BIN}" -m pytest -q FixedFeedbackSFT/tests/test_layout.py

while IFS= read -r -d '' script; do
  bash -n "${script}"
done < <(find FixedFeedbackSFT -path '*/commands/*.sh' -type f -print0)

echo "FixedFeedbackSFT layout and shell syntax checks passed."
