#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
CONFIG="${1:-${HERE}/configs/bench2drive_base_scratch.yaml}"
XML_PYTHON="${XML_PYTHON:-/home/guest3/miniconda3/envs/xml/bin/python}"
TRAINING_CUDA="${TRAINING_CUDA:-2}"

cleanup() {
  "${HERE}/stop_carla_bridge_rfl.sh" || true
}
trap cleanup EXIT INT TERM

"${HERE}/stop_carla_bridge_rfl.sh" || true
"${HERE}/start_carla_bridge_rfl.sh"
cd "${REPO_ROOT}"
CUDA_VISIBLE_DEVICES="${TRAINING_CUDA}" "${XML_PYTHON}" -m \
  experiments.qwen3_vl_embedding_2b_bdd100k_bench2drive_e2e_optical_moe16 \
  --config "${CONFIG}" \
  --phase sac_train
