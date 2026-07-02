from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from .utils import cuda_synchronize


@dataclass(frozen=True)
class BenchmarkResult:
    first_batch_latency_sec: float
    steady_state_latency_ms_per_image: float
    steady_state_images_per_second: float
    warmup_time_sec: float = 0.0
    measurement_time_sec: float = 0.0
    measured_images: int = 0
    total_time_sec: float = 0.0


def benchmark_callable(
    operation: Callable[[], int],
    device: torch.device,
    warmup_batches: int,
    benchmark_batches: int,
) -> BenchmarkResult:
    """Benchmark an operation returning the number of images it processed."""

    cuda_synchronize(device)
    first_start = time.perf_counter()
    first_count = operation()
    cuda_synchronize(device)
    first_latency = time.perf_counter() - first_start
    if first_count <= 0:
        raise ValueError("Benchmark operation must process at least one image.")

    cuda_synchronize(device)
    warmup_start = time.perf_counter()
    for _ in range(warmup_batches):
        operation()
    cuda_synchronize(device)
    warmup_elapsed = time.perf_counter() - warmup_start

    if benchmark_batches <= 0:
        return BenchmarkResult(
            first_latency,
            0.0,
            0.0,
            warmup_time_sec=warmup_elapsed,
            total_time_sec=first_latency + warmup_elapsed,
        )
    image_count = 0
    cuda_synchronize(device)
    steady_start = time.perf_counter()
    for _ in range(benchmark_batches):
        image_count += operation()
    cuda_synchronize(device)
    steady_elapsed = time.perf_counter() - steady_start
    images_per_second = image_count / steady_elapsed if steady_elapsed > 0 else 0.0
    latency_ms = 1000.0 / images_per_second if images_per_second > 0 else 0.0
    return BenchmarkResult(
        first_latency,
        latency_ms,
        images_per_second,
        warmup_time_sec=warmup_elapsed,
        measurement_time_sec=steady_elapsed,
        measured_images=image_count,
        total_time_sec=first_latency + warmup_elapsed + steady_elapsed,
    )
