"""Shared evidence helpers for formal frozen-Qwen GPU baselines.

The GPU power sampler deliberately uses the already installed ``nvidia-smi``
binary.  This avoids adding a Python/NVML dependency to a formal run.  Samples
are tagged by a caller-controlled phase, so preprocessing gaps are not mixed
into the model-active power average.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


RTX5090D_POWER_LIMIT_W = 575.0


def validate_cuda_device(expected_name_substring: str | None = None) -> str:
    """Return the selected CUDA device name and enforce an optional contract."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for a formal GPU benchmark")
    actual = torch.cuda.get_device_name(0)
    expected = (expected_name_substring or "").strip()
    if expected and expected.casefold() not in actual.casefold():
        raise RuntimeError(
            f"Formal benchmark requires a GPU name containing {expected!r}, got {actual!r}"
        )
    return actual


def nvidia_smi_gpu_id() -> str:
    """Return the physical GPU identifier selected by CUDA_VISIBLE_DEVICES.

    PyTorch renumbers a single visible physical GPU to ``cuda:0`` while
    ``nvidia-smi --id`` keeps the host's physical numbering.  Formal power
    evidence must therefore use the first CUDA-visible physical identifier.
    """

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return "0"
    selected = visible.split(",", 1)[0].strip()
    if not selected or selected == "-1":
        raise RuntimeError(f"Invalid CUDA_VISIBLE_DEVICES={visible!r}")
    return selected


def gpu_power_limit_w(gpu_index: int | str | None = None) -> float:
    """Read the active board power limit instead of assuming a GPU model."""

    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={nvidia_smi_gpu_id() if gpu_index is None else gpu_index}",
            "--query-gpu=power.limit",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        return float(result.stdout.strip().splitlines()[0])
    except (IndexError, ValueError) as error:
        raise RuntimeError(
            f"Could not parse nvidia-smi power limit: {result.stdout!r}"
        ) from error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def summarize(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty sequence")
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std()),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


@dataclass(frozen=True)
class PowerSample:
    host_time_unix: float
    host_monotonic_s: float
    phase: str
    watts: float


class NvidiaSmiPowerSampler:
    """Read board power at 20 Hz and retain only explicitly tagged windows."""

    def __init__(self, gpu_index: int | str | None = None, interval_ms: int = 50) -> None:
        if interval_ms > 50:
            raise ValueError("Formal power sampling must run at least at 20 Hz")
        self.gpu_index = nvidia_smi_gpu_id() if gpu_index is None else str(gpu_index)
        self.interval_ms = int(interval_ms)
        self._phase: str | None = None
        self._phase_lock = threading.Lock()
        self._samples: list[PowerSample] = []
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("Power sampler is already running")
        command = [
            "nvidia-smi",
            f"--id={self.gpu_index}",
            "--query-gpu=power.draw",
            "--format=csv,noheader,nounits",
            f"--loop-ms={self.interval_ms}",
        ]
        self._process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        try:
            for line in self._process.stdout:
                value = line.strip()
                if not value:
                    continue
                try:
                    watts = float(value)
                except ValueError:
                    continue
                with self._phase_lock:
                    phase = self._phase
                if phase is not None:
                    self._samples.append(
                        PowerSample(time.time(), time.monotonic(), phase, watts)
                    )
        except BaseException as error:  # surfaced by stop(), including I/O errors
            self._error = error

    def set_phase(self, phase: str | None) -> None:
        with self._phase_lock:
            self._phase = phase

    def stop(self) -> list[PowerSample]:
        process = self._process
        if process is None:
            return list(self._samples)
        self.set_phase(None)
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
        if self._thread is not None:
            self._thread.join(timeout=3)
        stderr = process.stderr.read().strip() if process.stderr is not None else ""
        self._process = None
        if self._error is not None:
            raise RuntimeError("nvidia-smi power sampler failed") from self._error
        if process.returncode not in (0, -15, 1) and stderr:
            raise RuntimeError(f"nvidia-smi power sampler failed: {stderr}")
        return list(self._samples)

    def __enter__(self) -> "NvidiaSmiPowerSampler":
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()


class FirstBlockTimer:
    """Time from the first invocation of a chosen Transformer block to output."""

    def __init__(self, first_block: torch.nn.Module) -> None:
        self._state: dict[str, Any] = {}
        self._handle = first_block.register_forward_pre_hook(self._start)

    def _start(self, _module: torch.nn.Module, args: tuple[Any, ...]) -> None:
        if self._state.get("calls", 0) == 0:
            self._state["start_event"].record()
            self._state["host_started"] = time.perf_counter()
            if args and isinstance(args[0], torch.Tensor):
                self._state["input_shape"] = list(args[0].shape)
        self._state["calls"] = self._state.get("calls", 0) + 1

    def reset(self) -> None:
        self._state = {
            "start_event": torch.cuda.Event(enable_timing=True),
            "end_event": torch.cuda.Event(enable_timing=True),
            "calls": 0,
        }

    def finish(self) -> dict[str, Any]:
        if self._state.get("calls", 0) < 1:
            raise RuntimeError("The configured first Transformer block was not called")
        self._state["end_event"].record()
        self._state["end_event"].synchronize()
        return {
            "cuda_ms": float(
                self._state["start_event"].elapsed_time(self._state["end_event"])
            ),
            "host_ms": 1000.0
            * (time.perf_counter() - self._state["host_started"]),
            "first_block_calls": int(self._state["calls"]),
            "first_block_input_shape": self._state.get("input_shape"),
        }

    def close(self) -> None:
        self._handle.remove()


def save_power_samples(path: Path, samples: Sequence[PowerSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["host_time_unix", "host_monotonic_s", "phase", "watts"])
        for sample in samples:
            writer.writerow(
                [sample.host_time_unix, sample.host_monotonic_s, sample.phase, sample.watts]
            )


def power_report(
    samples: Sequence[PowerSample],
    latencies_ms: Sequence[float],
    *,
    power_limit_w: float | None = None,
) -> dict[str, Any]:
    idle = [sample.watts for sample in samples if sample.phase == "idle"]
    active = [sample.watts for sample in samples if sample.phase.startswith("active:")]
    if not idle:
        raise RuntimeError("No idle power samples were captured")
    if not active:
        raise RuntimeError("No model-active power samples were captured")
    idle_mean = float(np.mean(idle))
    active_mean = float(np.mean(active))
    latency = summarize(latencies_ms)
    rated_power_w = gpu_power_limit_w() if power_limit_w is None else float(power_limit_w)
    return {
        "sampler": "nvidia-smi board power.draw",
        "sampling_interval_ms": 50,
        "idle_samples": len(idle),
        "active_samples": len(active),
        "idle_mean_w": idle_mean,
        "active_mean_w": active_mean,
        "active_peak_w": float(np.max(active)),
        "rated_power_limit_w": rated_power_w,
        "measured_active_energy_j_per_sample": active_mean * latency["mean"] / 1000.0,
        "idle_subtracted_energy_j_per_sample": max(0.0, active_mean - idle_mean)
        * latency["mean"]
        / 1000.0,
        "rated_upper_bound_energy_j_per_sample": rated_power_w
        * latency["mean"]
        / 1000.0,
        "energy_basis": (
            "board power sampled only while model inference was active; energy uses "
            "the measured first-block-to-task-output mean latency"
        ),
    }


def environment_report(*, power_limit_w: float | None = None) -> dict[str, Any]:
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu,
        "gpu_index": 0,
        "nvidia_smi_physical_gpu_id": nvidia_smi_gpu_id() if gpu is not None else None,
        "dtype": "bfloat16",
        "batch_size": 1,
        "rated_power_limit_w": (
            gpu_power_limit_w() if gpu is not None and power_limit_w is None else power_limit_w
        ),
    }


__all__ = [
    "NvidiaSmiPowerSampler",
    "FirstBlockTimer",
    "PowerSample",
    "RTX5090D_POWER_LIMIT_W",
    "environment_report",
    "gpu_power_limit_w",
    "nvidia_smi_gpu_id",
    "power_report",
    "save_power_samples",
    "sha256_file",
    "summarize",
    "validate_cuda_device",
    "write_json",
]
