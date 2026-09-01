#!/usr/bin/env bash
set -euo pipefail

# Register an existing ImageNet-1K cache in a linked Git worktree without
# copying or downloading the 312 GiB dataset.  The target is always the
# current 2026OpticsMoE checkout's data/imagenet1k path.

REPOSITORY_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPOSITORY_ROOT}"

IMAGENET1K_SOURCE="${IMAGENET1K_SOURCE:?Set the existing ImageNet-1K directory}"
SOURCE="$(readlink -f "${IMAGENET1K_SOURCE}")"
TARGET="${REPOSITORY_ROOT}/data/imagenet1k"

if [[ ! -d "${SOURCE}/huggingface_cache" ]]; then
  echo "Source does not contain huggingface_cache: ${SOURCE}" >&2
  exit 1
fi

if [[ -L "${TARGET}" ]]; then
  if [[ "$(readlink -f "${TARGET}")" != "${SOURCE}" ]]; then
    echo "ImageNet target is already linked to a different source: ${TARGET}" >&2
    exit 1
  fi
  echo "ImageNet cache link already registered: ${TARGET} -> ${SOURCE}"
  exit 0
fi

if [[ -d "${TARGET}/huggingface_cache" ]]; then
  rmdir "${TARGET}/huggingface_cache" 2>/dev/null || {
    echo "Refusing to replace a non-empty cache directory: ${TARGET}/huggingface_cache" >&2
    exit 1
  }
fi
if [[ -d "${TARGET}" ]]; then
  rmdir "${TARGET}" 2>/dev/null || {
    echo "Refusing to replace a non-empty ImageNet directory: ${TARGET}" >&2
    exit 1
  }
elif [[ -e "${TARGET}" ]]; then
  echo "Refusing to replace a non-directory ImageNet target: ${TARGET}" >&2
  exit 1
fi

mkdir -p "$(dirname "${TARGET}")"
ln -s "${SOURCE}" "${TARGET}"
echo "Registered ImageNet cache: ${TARGET} -> ${SOURCE}"
