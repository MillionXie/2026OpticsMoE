#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_progressive_growth_common.sh"

TARGET_DEPTH="${TARGET_DEPTH:?Set TARGET_DEPTH=32, 64, or 100 explicitly}"
select_progressive_growth_stage "${TARGET_DEPTH}"
render_or_verify_progressive_config "${TARGET_DEPTH}"
echo "Guarded config is ready: ${TARGET_CONFIG}"
