#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)"
bundle="${LG2_BUNDLE_ROOT:-$repo/t12_scene_full70_8step_20261102}"
cd "$repo"
exec "$bundle/.venv/bin/python" -m LightGenV2.tasks.t12_text_to_image.scene_portable_inference --bundle "$bundle" "$@"
