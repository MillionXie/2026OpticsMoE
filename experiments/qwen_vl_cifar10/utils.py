from __future__ import annotations

import json
import os
import platform
import random
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        warnings.warn(
            "CUDA was requested but is unavailable; falling back to CPU.", stacklevel=2
        )
        return torch.device("cpu")
    return torch.device(requested)


def resolve_dtype(name: str, device: torch.device) -> tuple[torch.dtype, str]:
    aliases = {
        "float32": (torch.float32, "float32"),
        "fp32": (torch.float32, "float32"),
        "float16": (torch.float16, "float16"),
        "fp16": (torch.float16, "float16"),
        "bfloat16": (torch.bfloat16, "bfloat16"),
        "bf16": (torch.bfloat16, "bfloat16"),
    }
    dtype, canonical = aliases[name]
    if device.type == "cpu" and dtype != torch.float32:
        warnings.warn(
            f"{canonical} was requested on CPU; using float32 for broader operator support.",
            stacklevel=2,
        )
        return torch.float32, "float32"
    return dtype, canonical


def cuda_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        for index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(index)


def reset_cuda_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        # PyTorch 2.6 can reject reset_peak_memory_stats before the CUDA
        # context for a visible device has been initialized. Synchronization
        # initializes each context and establishes a clean timing boundary.
        cuda_synchronize(device)
        for index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(index)


def cuda_peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    values = cuda_peak_memory_by_device_mb(device)
    return max(values.values(), default=0.0)


def cuda_peak_memory_by_device_mb(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    return {
        f"cuda:{index}": float(torch.cuda.max_memory_allocated(index) / (1024**2))
        for index in range(torch.cuda.device_count())
    }


def runtime_metadata(device: torch.device) -> dict[str, Any]:
    gpu_name = None
    gpu_total_memory_mb = 0.0
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        gpu_name = properties.name
        gpu_total_memory_mb = float(properties.total_memory / (1024**2))
    visible_gpus = []
    if device.type == "cuda":
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            visible_gpus.append(
                {
                    "device": f"cuda:{index}",
                    "name": properties.name,
                    "total_memory_mb": float(properties.total_memory / (1024**2)),
                }
            )
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version()
        if torch.backends.cudnn.is_available()
        else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "visible_cuda_device_count": torch.cuda.device_count()
        if torch.cuda.is_available()
        else 0,
        "runtime_device": str(device),
        "gpu_name": gpu_name,
        "gpu_total_memory_mb": gpu_total_memory_mb,
        "visible_gpus": visible_gpus,
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, default=_json_default)
        handle.write("\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def require_dependency(module_name: str, install_name: str | None = None) -> None:
    try:
        __import__(module_name)
    except ImportError as exc:
        package = install_name or module_name
        raise RuntimeError(
            f"Missing optional dependency '{module_name}'. Install it with `pip install {package}` "
            "or install this experiment's requirements.txt."
        ) from exc
