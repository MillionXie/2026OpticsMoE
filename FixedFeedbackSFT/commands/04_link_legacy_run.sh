#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 <absolute-source-run-dir> <project-name> <run-name>" >&2
  exit 2
fi

SOURCE_RUN="$(realpath "$1")"
PROJECT_NAME="$2"
RUN_NAME="$3"
REPOSITORY_ROOT="$(git rev-parse --show-toplevel)"

KNOWN_PROJECT=false
for candidate in \
  d2nn_cifar100c10_fixed_feedback_20stage400 \
  d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400 \
  d2nn_cifar10_high_performance_optical_backbone \
  qwen3_vl_patch_stem_8stage_optical_imagenet_backbone \
  qwen3_vl_patch_stem_8stage_slim_mixer_imagenet_backbone \
  qwen3_vl_patch_stem_8stage_dual_scale_optical_imagenet_backbone \
  qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone \
  qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa \
  qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone
do
  if [[ "${PROJECT_NAME}" == "${candidate}" ]]; then
    KNOWN_PROJECT=true
    break
  fi
done
if [[ "${KNOWN_PROJECT}" != true ]]; then
  echo "unknown FixedFeedbackSFT project: ${PROJECT_NAME}" >&2
  exit 2
fi

if [[ ! -d "${SOURCE_RUN}" ]]; then
  echo "source run directory does not exist: ${SOURCE_RUN}" >&2
  exit 2
fi
if [[ "${RUN_NAME}" == */* || -z "${RUN_NAME}" ]]; then
  echo "run name must be one path component" >&2
  exit 2
fi

TARGET_PARENT="${REPOSITORY_ROOT}/FixedFeedbackSFT/runs/${PROJECT_NAME}"
TARGET="${TARGET_PARENT}/${RUN_NAME}"
mkdir -p "${TARGET_PARENT}"

if [[ -L "${TARGET}" ]]; then
  if [[ "$(realpath "${TARGET}")" == "${SOURCE_RUN}" ]]; then
    echo "run link already points to ${SOURCE_RUN}"
    exit 0
  fi
  echo "refusing to replace existing run link: ${TARGET}" >&2
  exit 1
fi
if [[ -e "${TARGET}" ]]; then
  echo "refusing to replace existing path: ${TARGET}" >&2
  exit 1
fi

ln -s "${SOURCE_RUN}" "${TARGET}"
echo "linked ${TARGET} -> ${SOURCE_RUN}"
