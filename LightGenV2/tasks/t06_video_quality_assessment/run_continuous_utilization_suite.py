"""Fixed old three-phase native utilization protocol, with bounded small batches.

No Graph, no training, no cost replacement. All models get the same three
30-second phases in the same order; Qwen B1 gets no extra first-200 audit.
Report the predeclared third-phase seconds 5--30, retaining all other data.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from . import audit_utilization_execution as audit
from . import evaluate_temporal_baseline_batches as frozen
from . import reproduce_temporal_baseline as reproduction
from . import run_temporal_telemetry_suite as prior

TASK = Path(__file__).resolve().parent
REPO = TASK.parents[2]
INITIAL = {"qwen": 8, "clip": 9, "yolo": 3, "deepseek": 4}
LARGER = {"qwen": 10, "clip": 10, "yolo": 4, "deepseek": 8}
TARGET_PERCENT = 98.0
SATURATED_PERCENT = 99.5
MAX_GROUPS = 16


def next_batch(baseline, values):
    first = INITIAL[baseline]
    if first not in values:
        return first
    selected = first
    if values[first] < TARGET_PERCENT:
        selected = LARGER[baseline]
        if selected not in values:
            return selected
    if values[selected] >= SATURATED_PERCENT and selected-1 not in values:
        return selected-1
    return None


def run(args):
    import fcntl
    reproduction.require_new_output(args.output)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != frozen.A100_UUID:
        raise RuntimeError("Approved A100 required")
    if frozen.sha256(args.manifest) != frozen.MANIFEST_SHA:
        raise ValueError("Original manifest required")
    # Parent holds the shared lock; children use the existing old measurement.
    with (TASK / "runs/simulation/.temporal_telemetry_a100.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for i in range(3):
            reproduction.cuda_guard()
            if i < 2:
                time.sleep(15)
        args.output.mkdir(parents=True, exist_ok=False)
        files = (Path(__file__), Path(audit.__file__), Path(frozen.__file__), TASK / "batch_telemetry.py")
        hashes = {p.name: frozen.sha256(p) for p in files}
        state = dict(status="running", pid=os.getpid(), started_unix=time.time(), deadline_unix=args.deadline,
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            source_sha256=hashes, completed=[], initial_batches=INITIAL, larger_batches=LARGER,
            target_mean_percent=TARGET_PERCENT, saturation_mean_percent=SATURATED_PERCENT,
            max_groups=MAX_GROUPS,
            protocol="Fixed legacy_sync_cpu then sync_gpu_output then queued_gpu_output, each30s; primary third-phase seconds5--30, no Graph or extra Qwen first200 audit")

        def save():
            frozen.write_json(args.output / "suite_status.json", state)

        def dispatch(baseline, batch):
            if len(state["completed"]) >= MAX_GROUPS or time.time() >= args.deadline:
                raise RuntimeError("Bounded coordinated window exhausted")
            reproduction.cuda_guard()
            if hashes != {p.name: frozen.sha256(p) for p in files}:
                raise RuntimeError("Running measurement source changed")
            name = f"{baseline}_batch{batch:02d}"
            state["phase"] = name
            save()
            command = prior.child_command(baseline, batch, args.manifest, args.output / name)
            command[command.index(prior.__package__ + ".evaluate_temporal_baseline_batches")] = __package__ + ".audit_utilization_execution"
            command.remove("--telemetry")
            command.extend(["--phase", "gpu", "--utilization-only"])
            frozen.write_json(args.output / f"{name}.command.json", command)
            with (args.output / f"{name}.log").open("x") as stream:
                subprocess.run(command, cwd=REPO, check=True, stdout=stream, stderr=subprocess.STDOUT,
                               timeout=min(600, max(1, args.deadline-time.time())))
            report = json.loads((args.output / name / "report.json").read_text())
            if (report["status"] != "complete" or report["preceding_forwards"] != 0
                    or [m["mode"] for m in report["modes"]] != list(audit.MODES)):
                raise RuntimeError("Uniform continuous protocol failed")
            value = report["modes"][2]["windows"][1]["utilization_percent"]["mean"]
            state["completed"].append(dict(name=name, baseline=baseline, batch=batch, mean_utilization_percent=value))
            save()
            return value

        save()
        try:
            for baseline in INITIAL:
                dispatch(baseline, 1)
                values = {}
                while (batch := next_batch(baseline, values)) is not None:
                    values[batch] = dispatch(baseline, batch)
            reproduction.cuda_guard()
            state.update(status="complete", completed_unix=time.time())
        except BaseException as error:
            state.update(status="failed", error=repr(error))
            raise
        finally:
            save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    args.manifest, args.output = args.manifest.resolve(), args.output.resolve()
    if not 0 < args.deadline-time.time() <= 2400:
        raise ValueError("Explicit coordinated deadline within40min required")
    run(args)


if __name__ == "__main__":
    main()
