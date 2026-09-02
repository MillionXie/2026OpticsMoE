#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

CONFIG="${PROJECT}/configs/plumbing_smoke_imagenet1k_21841_gpu5.yaml"
GPU="${IN22K_SMOKE_GPU:-5}"
ACTION="${IN22K_SMOKE_ACTION:-fresh}"

# Asset/cache/disk checks happen before GPU inspection and before a log/output
# directory is created.  This smoke is explicitly not an ImageNet-22K result.
preflight_cpu "${CONFIG}"
require_idle_gpu "${GPU}"
UUID="$(gpu_uuid "${GPU}")"

ARGS=(--config "${CONFIG}")
case "${ACTION}" in
  fresh) ;;
  resume) ARGS+=(--resume) ;;
  *) echo "IN22K_SMOKE_ACTION must be fresh or resume" >&2; exit 2 ;;
esac

echo "PLUMBING ONLY: ImageNet-1K images, random 21841-way head, no performance claim"
CUDA_VISIBLE_DEVICES="${UUID}" "${PYTHON_BIN}" -m "${MODULE}" "${ARGS[@]}"
