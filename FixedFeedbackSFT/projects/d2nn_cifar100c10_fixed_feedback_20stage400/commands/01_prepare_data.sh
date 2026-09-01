#!/usr/bin/env bash
set -euo pipefail
python -m experiments.d2nn_cifar100c10_fixed_feedback_20stage400 \
  --config FixedFeedbackSFT/projects/d2nn_cifar100c10_fixed_feedback_20stage400/configs/main.yaml \
  --phase prepare_data
