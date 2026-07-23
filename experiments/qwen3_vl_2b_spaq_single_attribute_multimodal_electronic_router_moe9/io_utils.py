from __future__ import annotations

import csv
import json
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch


def configure_cpu_runtime(cpu_threads: int, cpu_interop_threads: int) -> dict[str, int]:
    """Bound PyTorch CPU pools used by cached-tensor collation and metrics.

    Student training intentionally keeps its cache-backed DataLoader in the
    main process: spawning workers would duplicate the multi-gigabyte shard
    LRU.  Explicit thread limits prevent tiny CPU tensor operations from
    waking every core on large servers.
    """
    torch.set_num_threads(int(cpu_threads))
    requested_interop = int(cpu_interop_threads)
    try:
        torch.set_num_interop_threads(requested_interop)
    except RuntimeError:
        # PyTorch permits setting the inter-op pool only before parallel work
        # starts. Repeated in-process test invocations are safe when the
        # already configured value is identical.
        if torch.get_num_interop_threads() != requested_interop:
            raise
    return {
        "torch_cpu_threads": torch.get_num_threads(),
        "torch_cpu_interop_threads": torch.get_num_interop_threads(),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def resolve_dtype(value: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[value]


def runtime_metadata(device: torch.device) -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_cpu_threads": torch.get_num_threads(),
        "torch_cpu_interop_threads": torch.get_num_interop_threads(),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    raise TypeError(f"Cannot serialize {type(value).__name__}")
