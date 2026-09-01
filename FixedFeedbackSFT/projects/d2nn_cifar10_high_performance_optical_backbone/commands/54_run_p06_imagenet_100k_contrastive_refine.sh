#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

PHYSICAL_GPU_INDICES="${PHYSICAL_GPU_INDICES:-2,4}"
IFS=',' read -r -a gpu_indices <<< "${PHYSICAL_GPU_INDICES}"
gpu_uuids=()
for gpu_index in "${gpu_indices[@]}"; do
  gpu_uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | sed -n "$((gpu_index + 1))p")"
  : "${gpu_uuid:?Could not resolve physical GPU ${gpu_index}}"
  gpu_uuids+=("${gpu_uuid}")
done
export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${gpu_uuids[*]}")"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

TORCHRUN_BIN="$(dirname "${PYTHON_BIN}")/torchrun"
"${TORCHRUN_BIN}" \
  --standalone \
  --nproc_per_node="${#gpu_indices[@]}" \
  -m experiments.d2nn_cifar10_high_performance_optical_backbone.general_backbone_pretraining \
  --config FixedFeedbackSFT/projects/d2nn_cifar10_high_performance_optical_backbone/configs/p06_imagenet_100k_contrastive_refine.yaml \
  --resume
