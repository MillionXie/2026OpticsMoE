"""Unified utilization-only resident inference, without CUDA Graph or training.

Prepare the first B TEST videos once; serial same-stream forwards retain B
videos per invocation. Eight queued invocations are not an enlarged batch.
Five seconds of real-forward warmup precede an independent 30-second window.
Historical paper latency, energy and quality are never replaced by this run.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import threading
import time

from . import audit_utilization_execution as execution
from . import batch_telemetry as telemetry
from . import evaluate_temporal_baseline_batches as frozen

TASK = Path(__file__).resolve().parent
WARMUP_S = 5
MEASUREMENT_S = 30
QUEUE_DEPTH = 8


def summarize(samples, start, end):
    """Exact predeclared interval, retaining low and zero utilization samples."""
    selected = [s for s in samples if start <= s.host_monotonic_s < end]
    if not selected:
        raise RuntimeError("No samples in the measured interval")
    result = telemetry.telemetry_report(selected, prefix="")
    result.update(start_monotonic_s=start, end_monotonic_s=end,
                  duration_s=end-start,
                  min_gpu_utilization_percent=min(s.utilization_percent for s in selected),
                  median_gpu_utilization_percent=statistics.median(s.utilization_percent for s in selected))
    return result


class ForeignGuard:
    """Read-only periodic process audit, not a workload or reservation."""
    def __init__(self):
        self.stopped = threading.Event()
        self.records = []
        self.error = None

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stopped.is_set():
            try:
                self.records.append(dict(monotonic_s=time.monotonic(), processes=frozen.process_audit()))
            except BaseException as error:
                self.error = error
                break
            self.stopped.wait(1)

    def check(self):
        if self.error:
            raise RuntimeError("Concurrent A100 process detected; exclude this run") from self.error

    def stop(self):
        self.stopped.set()
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError("Process audit did not finish")


def run(args):
    import fcntl
    from . import reproduce_temporal_baseline as reproduction
    from LightGenV2.common.baseline_measurement import gpu_power_limit_w, validate_cuda_device

    reproduction.require_new_output(args.output)
    frozen.preflight_files(args)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != frozen.A100_UUID:
        raise RuntimeError("Only the approved physical A100 is allowed")
    with (TASK / "runs/simulation/.temporal_telemetry_a100.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Three bounded idle observations before CUDA initialization; never wait
        # indefinitely or consume GPU work while another task is using the card.
        for i in range(3):
            reproduction.cuda_guard()
            if i < 2:
                time.sleep(15)
        if abs(gpu_power_limit_w()-250) > 0.1:
            raise RuntimeError("A100 power limit must remain 250 W")
        import torch
        args.output.mkdir(parents=True, exist_ok=False)
        identity = dict(status="loading", pid=os.getpid(), baseline=args.baseline, batch=args.batch,
            gpu_uuid=frozen.A100_UUID, checkpoint_sha256=frozen.sha256(args.checkpoint),
            manifest_sha256=frozen.sha256(args.manifest), command=sys.argv,
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            source_sha256={p.name: frozen.sha256(p) for p in
                (Path(__file__), Path(execution.__file__), Path(frozen.__file__), Path(telemetry.__file__))},
            protocol=dict(warmup_s=WARMUP_S, measurement_minimum_s=MEASUREMENT_S,
                queue_depth=QUEUE_DEPTH, cuda_graph=False, streams=1,
                input_preparation="first B TEST videos prepared and transferred once",
                workload="serial native full forwards with GPU MOS, synchronize after eight invocations",
                selection="all samples in exact post-warmup measurement interval; no window search",
                utilization_only=True, historical_costs_unchanged=True),
            environment=dict(python=platform.python_version(), torch=torch.__version__,
                             cuda=torch.version.cuda, gpu=validate_cuda_device("A100")))
        frozen.write_json(args.output / "status.json", identity)
        timer, handles, sampler, guard = None, [], None, None
        samples, groups = [], []
        try:
            with torch.inference_mode():
                model, head, rows, prepare, transfer, forward, timer, handles = frozen.load_runtime(args)
                if len(rows) != 558 or any(p.requires_grad for m in (model, head) for p in m.parameters()):
                    raise RuntimeError("Full TEST identity and frozen model required")
                from .diagnose_batch_numerics import input_records
                cpu, positions = prepare(rows[:args.batch])
                fingerprints = input_records(cpu, args.batch)
                resident = transfer(cpu)
                torch.cuda.synchronize()
                reference, _, _ = forward(resident)
                timer.close()
                compute = execution.make_gpu_forward(args, model, head, forward)
                initial = compute(resident).detach().cpu().tolist()
                delta = max(abs(a-b) for a, b in zip(reference, initial))
                if delta > 0.0002:
                    raise RuntimeError(f"GPU-output implementation changed MOS: {delta}")
                frozen.process_audit()
                guard = ForeignGuard()
                guard.start()
                sampler = telemetry.TelemetrySampler(frozen.A100_UUID)
                sampler.start()

                def phase(name, minimum_duration):
                    sampler.set_phase(name)
                    begin = time.monotonic()
                    while time.monotonic()-begin < minimum_duration:
                        guard.check()
                        tick = time.monotonic()
                        for _ in range(QUEUE_DEPTH):
                            last = compute(resident)
                        torch.cuda.synchronize()
                        groups.append(dict(phase=name, start_monotonic_s=tick,
                                           end_monotonic_s=time.monotonic(), forwards=QUEUE_DEPTH))
                    end = time.monotonic()
                    guard.check()
                    return begin, end, last

                frozen.write_json(args.output / "status.json", {**identity, "status": "warmup"})
                warm_start, warm_end, _ = phase("warmup", WARMUP_S)
                # No file writes, H2D or CPU MOS copies inside either loop.
                measured_start, measured_end, last = phase("measured", MEASUREMENT_S)
                sampler.set_phase("after")
                samples = sampler.stop()
                sampler = None
                guard.stop()
                guard.check()
                final = last.detach().cpu().tolist()
                final_delta = max(abs(a-b) for a, b in zip(reference, final))
                if final_delta > 0.0002:
                    raise RuntimeError("Repeated resident inference changed output")
                end_audit = frozen.process_audit()
                if abs(gpu_power_limit_w()-250) > 0.1:
                    raise RuntimeError("A100 power limit changed")
                measured = [g for g in groups if g["phase"] == "measured"]
                report = {**identity, "status": "complete", "completed_unix": time.time(),
                    "sample_ids": [r["sample_id"] for r in rows[:args.batch]],
                    "frame_positions": positions, "input_fingerprints": fingerprints,
                    "reference_mos": reference, "final_mos": final,
                    "initial_max_abs_mos_vs_native": delta,
                    "final_max_abs_mos_vs_native": final_delta,
                    "warmup": summarize(samples, warm_start, warm_end),
                    "measurement": summarize(samples, measured_start, measured_end),
                    "measured_forward_invocations": sum(g["forwards"] for g in measured),
                    "distinct_test_videos": args.batch,
                    "process_audit_after": end_audit,
                    "cost_note": "Utilization-only resident load; no replacement latency, energy, SRCC or old-paper cost calculation."}
                frozen.write_json(args.output / "report.json", report)
                frozen.write_json(args.output / "status.json", {**identity, "status": "complete"})
        except BaseException as error:
            frozen.write_json(args.output / "status.json", {**identity, "status": "failed", "error": repr(error)})
            raise
        finally:
            if sampler is not None:
                samples = sampler.stop()
            if guard is not None:
                guard.stop()
                frozen.write_json(args.output / "process_audit.json", guard.records)
            telemetry.save_samples(args.output / "telemetry.csv", samples)
            frozen.write_json(args.output / "forward_groups.json", groups)
            if timer is not None:
                timer.close()
            for handle in handles:
                handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", choices=tuple(frozen.CHECKPOINT_SHA), required=True)
    parser.add_argument("--batch", type=int, choices=(1, 2, 3, 4, 8, 9, 10), required=True)
    for name in ("manifest", "checkpoint", "model", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--reproduction-receipt", type=Path)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    run(args)


if __name__ == "__main__":
    main()
