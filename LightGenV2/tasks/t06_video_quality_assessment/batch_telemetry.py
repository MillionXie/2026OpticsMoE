"""Task-local paired power/utilization telemetry; no training or shared changes."""
from __future__ import annotations

import csv
import json
import math
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Sample:
    host_time_unix: float
    host_monotonic_s: float
    phase: str
    watts: float
    utilization_percent: float
    memory_used_mib: float


class TelemetrySampler:
    """50-ms polling does not change the driver's utilization averaging window."""
    def __init__(self, gpu_uuid: str, raw_path: Path | None = None):
        self.gpu_uuid = gpu_uuid
        self.phase = "unlabelled"
        self.lock = threading.Lock()
        self.samples = []
        self.process = None
        self.error = None
        self.raw_path = raw_path
        self.raw_stream = None
        self.stopping = False

    def start(self):
        self.stopping = False
        if self.raw_path is not None:
            self.raw_stream = self.raw_path.open("x", encoding="utf-8")
        self.process = subprocess.Popen([
            "nvidia-smi", f"--id={self.gpu_uuid}",
            "--query-gpu=power.draw,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits", "--loop-ms=50",
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                record = dict(host_time_unix=time.time(), host_monotonic_s=time.monotonic(),
                              phase=self.phase, raw_line=line.rstrip("\r\n"))
                try:
                    watts, utilization, memory = parse_telemetry_line(line)
                    with self.lock:
                        phase = self.phase
                    self.samples.append(Sample(time.time(), time.monotonic(), phase,
                                               watts, utilization, memory))
                    record["valid_numeric_row"] = True
                except BaseException as error:
                    record.update(valid_numeric_row=False, error=repr(error))
                    raise
                finally:
                    if self.raw_stream is not None:
                        self.raw_stream.write(json.dumps(record, ensure_ascii=True)+"\n")
                        self.raw_stream.flush()
            # A live sampler must not silently finish with an unexpected EOF.
            if not self.stopping:
                self.error = RuntimeError("Telemetry subprocess stdout ended")
        except BaseException as error:
            self.error = error

    def check(self):
        if self.error:
            raise RuntimeError(f"Telemetry reader failed: {self.error!r}") from self.error

    def set_phase(self, phase):
        with self.lock:
            self.phase = phase or "between_calls"

    def stop(self, raise_on_error=True):
        self.stopping = True
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
            self.thread.join(timeout=3)
            if self.thread.is_alive():
                self.error = RuntimeError("Telemetry reader did not stop")
            stderr = self.process.stderr.read().strip()
            code = self.process.returncode
            self.process = None
            if code not in (0, -15, 1) and stderr:
                self.error = RuntimeError(stderr)
        if self.raw_stream is not None:
            self.raw_stream.close()
            self.raw_stream = None
        if raise_on_error:
            self.check()
        return list(self.samples)


def parse_telemetry_line(line):
    """Reject malformed/N/A rows; never skip, impute or turn missing into zero."""
    fields = line.strip().split(",")
    if len(fields) != 3:
        raise ValueError(f"Expected3 telemetry fields: {line.strip()!r}")
    try:
        watts, utilization, memory = map(float, fields)
    except ValueError as error:
        raise ValueError(f"Non-numeric telemetry: {line.strip()!r}") from error
    if (not all(math.isfinite(x) for x in (watts, utilization, memory))
            or not 0 <= utilization <= 100 or watts <= 0 or memory < 0):
        raise ValueError(f"Invalid telemetry: {line.strip()!r}")
    return watts, utilization, memory


def save_samples(path: Path, samples):
    fields = tuple(Sample.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows([getattr(row, key) for key in fields] for row in samples)


def telemetry_report(samples, prefix="active:"):
    selected = [row for row in samples if row.phase.startswith(prefix)]
    if not selected:
        raise RuntimeError(f"No measured telemetry in {prefix!r} windows")
    return {
        "samples": len(selected), "phase_prefix": prefix,
        "mean_gpu_utilization_percent": statistics.mean(row.utilization_percent for row in selected),
        "peak_gpu_utilization_percent": max(row.utilization_percent for row in selected),
        "mean_memory_used_mib": statistics.mean(row.memory_used_mib for row in selected),
        "peak_memory_used_mib": max(row.memory_used_mib for row in selected),
        "mean_board_power_w": statistics.mean(row.watts for row in selected),
        "poll_interval_ms": 50,
        "utilization_definition": "nvidia-smi kernel-busy fraction over driver window, not SM occupancy or percent of peak FLOPS",
        "limitation": "Short tagged calls can contain utilization/power averaging from preceding activity; not calibrated instantaneous per-call counters",
    }


def inference_plan(total: int, batch: int):
    """Exactly 200 cost videos, including a partial boundary batch when needed."""
    if total != 558 or batch not in (1, 2, 3, 4, 8, 9, 10):
        raise ValueError("Unsupported workload")
    return [(start, min(start + batch, boundary))
            for begin, boundary in ((0, 200), (200, total))
            for start in range(begin, boundary, batch)]


def workload_cost(cost_rows, board_w: float, host_w=338.2):
    if sum(row[1] for row in cost_rows) != 200 or board_w <= 0:
        raise ValueError("Expected exactly 200 measured cost videos")
    time16 = sum(row[3] for row in cost_rows) * 16 / 200
    return {
        "latency_16_videos_ms": time16, "latency_per_video_ms": time16 / 16,
        "energy_16_videos_j": (host_w + board_w) * time16 / 1000,
        "energy_per_video_j": (host_w + board_w) * time16 / 16000,
        "throughput_videos_per_second": 16000 / time16,
        "host_proxy_w": host_w, "measured_active_board_w": board_w,
        "normalization": "sum of inference boundary times for exactly 200 videos * 16/200; includes partial cost batch",
    }
