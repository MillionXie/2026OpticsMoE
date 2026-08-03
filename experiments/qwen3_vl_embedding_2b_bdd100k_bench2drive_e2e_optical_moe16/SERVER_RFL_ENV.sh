#!/usr/bin/env bash
# Source this file on the configured 100right server before CARLA tools.
# Override any path in the shell when installing elsewhere.

export CARLA_ROOT="${CARLA_ROOT:-/DATA/DATA1/guest3/third_party/CARLA_0.9.15}"
export BENCH2DRIVE_ROOT="${BENCH2DRIVE_ROOT:-/DATA/DATA1/guest3/third_party/Bench2Drive}"
export CARLA_SERVER="${CARLA_SERVER:-${CARLA_ROOT}/CarlaUE4.sh}"
export SCENARIO_RUNNER_ROOT="${SCENARIO_RUNNER_ROOT:-${BENCH2DRIVE_ROOT}/scenario_runner}"
export LEADERBOARD_ROOT="${LEADERBOARD_ROOT:-${BENCH2DRIVE_ROOT}/leaderboard}"
export NVIDIA_GRAPHICS_ROOT="${NVIDIA_GRAPHICS_ROOT:-/DATA/DATA1/guest3/third_party/nvidia-550.144.03-user/extracted}"

CARLA_EGG="${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg"
export PYTHONPATH="${CARLA_EGG}:${CARLA_ROOT}/PythonAPI:${CARLA_ROOT}/PythonAPI/carla:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${BENCH2DRIVE_ROOT}:${PYTHONPATH:-}"

# This compute server originally had the NVIDIA kernel/CUDA driver but no
# Vulkan user-space ICD.  The user-local bundle is the exact same 550.144.03
# release. NVIDIA documents libEGL_nvidia as the ICD entry point for headless
# systems where an X11 client stack is not available.
if [[ -f "${NVIDIA_GRAPHICS_ROOT}/nvidia_headless_icd.json" ]]; then
  export LD_LIBRARY_PATH="${NVIDIA_GRAPHICS_ROOT}:${LD_LIBRARY_PATH:-}"
  export VK_ICD_FILENAMES="${NVIDIA_GRAPHICS_ROOT}/nvidia_headless_icd.json"
fi

if [[ ! -x "${CARLA_SERVER}" ]]; then
  echo "CARLA server is missing or not executable: ${CARLA_SERVER}" >&2
  return 1 2>/dev/null || exit 1
fi

echo "CARLA_ROOT=${CARLA_ROOT}"
echo "BENCH2DRIVE_ROOT=${BENCH2DRIVE_ROOT}"
echo "VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-system default}"
echo "Activate runtime with: conda activate RFL"
