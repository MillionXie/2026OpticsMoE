"""Bounded Fig. 5 utilization-only suite, reusing the completed Qwen B1.

No Graph, training, precision change, cost replacement or highest-run selection.
Initial high batches use historical native resident evidence. At most one larger
candidate and one adjacent lower batch are added per model. All trials retained.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from . import evaluate_temporal_baseline_batches as frozen
from . import measure_resident_utilization as resident
from . import run_temporal_telemetry_suite as prior
from . import reproduce_temporal_baseline as reproduction

TASK = Path(__file__).resolve().parent
REPO = TASK.parents[2]
TARGET_PERCENT = 98.0
SATURATED_PERCENT = 99.5
INITIAL = {"qwen": 16, "clip": 10, "yolo": 4, "deepseek": 8}
LARGER = {"qwen": 24, "clip": 11, "yolo": 5, "deepseek": 12}
MAX_NEW_GROUPS = 15  # 3 B1 + 4 models * at most 3 high-batch points


def next_batch(baseline, observed):
    """Predeclared finite decision tree, never select/repeat the same run."""
    initial = INITIAL[baseline]
    if initial not in observed:
        return initial
    first = observed[initial]
    candidate = initial
    if first < TARGET_PERCENT:
        candidate = LARGER[baseline]
        if candidate not in observed:
            return candidate
    if observed[candidate] >= SATURATED_PERCENT and candidate-1 not in observed:
        return candidate-1
    return None


def run(args):
    reproduction.require_new_output(args.output)
    reproduction.cuda_guard()
    if frozen.sha256(args.manifest) != frozen.MANIFEST_SHA:
        raise ValueError("Original manifest required")
    reused = json.loads((args.reuse_qwen_b1 / "report.json").read_text())
    if (reused.get("status") != "complete" or reused.get("baseline") != "qwen"
            or reused.get("batch") != 1 or reused.get("checkpoint_sha256") != frozen.CHECKPOINT_SHA["qwen"]
            or reused.get("manifest_sha256") != frozen.MANIFEST_SHA
            or reused.get("protocol", {}).get("cuda_graph") is not False
            or reused["protocol"].get("warmup_s") != resident.WARMUP_S
            or reused["protocol"].get("measurement_minimum_s") != resident.MEASUREMENT_S
            or reused["protocol"].get("queue_depth") != resident.QUEUE_DEPTH):
        raise ValueError("Qwen B1 reuse identity/protocol mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(resident.__file__), TASK / "audit_utilization_execution.py",
               Path(frozen.__file__), TASK / "batch_telemetry.py"]
    hashes = {p.name: frozen.sha256(p) for p in sources}
    state = dict(status="running", pid=os.getpid(), started_unix=time.time(),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        source_sha256=hashes, target_mean_percent=TARGET_PERCENT,
        saturation_mean_percent=SATURATED_PERCENT, initial_batches=INITIAL, larger_batches=LARGER,
        maximum_new_groups=MAX_NEW_GROUPS, completed=[],
        reused_qwen_b1=dict(path=str(args.reuse_qwen_b1),
                           report_sha256=frozen.sha256(args.reuse_qwen_b1 / "report.json")),
        note="Independent utilization only. Keep all attempted points; do not replace paper latency/energy/SRCC.")

    def save():
        frozen.write_json(args.output / "suite_status.json", state)

    def dispatch(baseline, batch):
        if time.time() >= args.deadline or len(state["completed"]) >= MAX_NEW_GROUPS:
            raise RuntimeError("Bounded suite budget exhausted; no automatic extension")
        if hashes != {p.name: frozen.sha256(p) for p in sources}:
            raise RuntimeError("Running measurement source changed")
        reproduction.cuda_guard()
        name = f"{baseline}_batch{batch:02d}"
        state["phase"] = name
        save()
        command = prior.child_command(baseline, batch, args.manifest, args.output / name)
        command[command.index(prior.__package__ + ".evaluate_temporal_baseline_batches")] = __package__ + ".measure_resident_utilization"
        command.remove("--telemetry")
        frozen.write_json(args.output / f"{name}.command.json", command)
        with (args.output / f"{name}.log").open("x") as stream:
            subprocess.run(command, cwd=REPO, check=True, stdout=stream, stderr=subprocess.STDOUT,
                           timeout=min(600, max(1, args.deadline-time.time())))
        report = json.loads((args.output / name / "report.json").read_text())
        if report["status"] != "complete":
            raise RuntimeError("Incomplete resident result")
        value = report["measurement"]["mean_gpu_utilization_percent"]
        if not 0 <= value <= 100:
            raise RuntimeError("Invalid driver GPU utilization")
        state["completed"].append(dict(name=name, baseline=baseline, batch=batch, mean_utilization_percent=value))
        save()
        return value

    save()
    try:
        for baseline in ("clip", "yolo", "deepseek"):
            dispatch(baseline, 1)
        for baseline in INITIAL:
            observed = {}
            while (batch := next_batch(baseline, observed)) is not None:
                observed[batch] = dispatch(baseline, batch)
        reproduction.cuda_guard()
        state.update(status="complete", completed_unix=time.time())
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        raise
    finally:
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "output", "reuse-qwen-b1"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    if not 0 < args.deadline-time.time() <= 3600:
        raise ValueError("Explicit deadline within one hour required")
    run(args)


if __name__ == "__main__":
    main()
