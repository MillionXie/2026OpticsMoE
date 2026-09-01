#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root.  This command never searches historical
# worktrees: P11, P13 and the frozen stem must already be registered as links
# below FixedFeedbackSFT/runs.  The outputs are committed atomically to
# FixedFeedbackSFT/runs/_assets/{8stage,16stage}.

REPOSITORY_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPOSITORY_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
exec "${PYTHON_BIN}" FixedFeedbackSFT/tools/freeze_backbone_assets.py "$@"
