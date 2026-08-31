#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/guest3/miniconda3/envs/xml/bin/python}"
EXPERIMENT="experiments/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone"
P11_EXPERIMENT="experiments/qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone"
STEM_CHECKPOINT="${STEM_CHECKPOINT:-experiments/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/assets/qwen3_vl_static_stem_224.pt}"
P11_CHECKPOINT="${P11_CHECKPOINT:-${P11_EXPERIMENT}/runs/p11_imagenet1k_pretrain_bs96_90e/checkpoints/backbone.pt}"
