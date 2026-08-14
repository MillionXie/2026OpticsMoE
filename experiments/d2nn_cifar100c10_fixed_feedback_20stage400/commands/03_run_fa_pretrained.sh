#!/usr/bin/env bash
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to one GPU index}"
python -m experiments.d2nn_cifar100c10_fixed_feedback_20stage400 \
  --config experiments/d2nn_cifar100c10_fixed_feedback_20stage400/configs/main.yaml \
  --phase run --method fa_pretrained
