"""User-authorized A100 paired frozen remeasurement, no retraining."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from . import evaluate_temporal_baseline_batches as frozen
from . import reproduce_temporal_baseline as reproduction

TASK = Path(__file__).resolve().parent
REPO = TASK.parents[2]
PREVIOUS = TASK / "runs/simulation/lgvq_temporal_reproduction_a100_20261013"
QWEN_MODEL = Path("/DATA/DATA1/guest3/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203")
QWEN_PT = Path("/DATA/DATA1/guest3/2026OpticsMoE_a100_formal/LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/qwen3vl_quality_tokens_r448_dataset_once_a100_20260907/checkpoints/frames4/best_checkpoint.pt")
QWEN_ENV = "/home/guest3/miniconda3/envs/xml/bin/python"


def child_command(baseline, batch, manifest, output, diagnostic=False):
    module = "diagnose_batch_numerics" if diagnostic else "evaluate_temporal_baseline_batches"
    env = QWEN_ENV if baseline == "qwen" else reproduction.ENV_PATHS[baseline]
    model = QWEN_MODEL if baseline == "qwen" else reproduction.MODEL_PATHS[baseline]
    pt = QWEN_PT if baseline == "qwen" else PREVIOUS / baseline / "checkpoints/temporal/best_checkpoint.pt"
    command = [env, "-u", "-m", __package__ + "." + module,
               "--baseline", baseline, "--manifest", str(manifest),
               "--model", str(model), "--checkpoint", str(pt), "--output", str(output)]
    if not diagnostic:
        command.extend(["--batch", str(batch), "--telemetry"])
    if baseline != "qwen":
        command.extend(["--reproduction-receipt", str(PREVIOUS / baseline / "reproduction_receipt.json")])
    return command


def run(args):
    import fcntl
    reproduction.require_new_output(args.output)
    reproduction.cuda_guard()
    files = [Path(__file__), Path(frozen.__file__), TASK / "batch_telemetry.py", TASK / "diagnose_batch_numerics.py"]
    source = {path.name: frozen.sha256(path) for path in files}
    with (TASK / "runs/simulation/.temporal_telemetry_a100.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.output.mkdir(parents=True, exist_ok=False)
        state = {"status": "running", "pid": os.getpid(), "gpu_uuid": frozen.A100_UUID,
                 "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                 "source_sha256": source, "completed": [], "diagnostic_errors": [],
                 "checkpoint_selection": "reuse existing frozen PT only, no training",
                 "conditional_rule": "CLIP B10 iff B9 15-second resident-input mean utilization <98%; no further batch scan"}
        def save():
            frozen.write_json(args.output / "suite_status.json", state)
        def dispatch(baseline, batch, diagnostic=False):
            reproduction.cuda_guard()
            if source != {path.name: frozen.sha256(path) for path in files}:
                raise RuntimeError("Suite source changed while running")
            name = f"diagnostic_{baseline}" if diagnostic else f"{baseline}_batch{batch:02d}"
            state["phase"] = name
            save()
            command = child_command(baseline, batch, args.manifest, args.output / name, diagnostic)
            with (args.output / f"{name}.command.json").open("x") as stream:
                json.dump(command, stream, indent=2)
            with (args.output / f"{name}.log").open("x") as stream:
                subprocess.run(command, cwd=REPO, check=True, stdout=stream, stderr=subprocess.STDOUT)
            state["completed"].append(name)
            save()
        save()
        try:
            # Requested extra points first; every point retains paired cost and performance.
            dispatch("yolo", 3)
            dispatch("clip", 9)
            clip9 = json.loads((args.output / "clip_batch09/report.json").read_text())
            utilization = clip9["sustained_resident_input"]["telemetry"]["mean_gpu_utilization_percent"]
            state["clip_b9_sustained_mean_utilization_percent"] = utilization
            state["clip_b10_required"] = utilization < 98
            save()
            if utilization < 98:
                dispatch("clip", 10)
            for baseline in ("qwen", "clip", "yolo", "deepseek"):
                for batch in (1, 2, 4, 8):
                    dispatch(baseline, batch)
            for baseline in ("qwen", "deepseek"):
                try:
                    dispatch(baseline, 1, diagnostic=True)
                except subprocess.CalledProcessError as error:
                    # Preserve partial input/intermediate evidence, no blind retry;
                    # failed precision probes do not erase valid cost measurements.
                    state["diagnostic_errors"].append({"baseline": baseline, "error": repr(error)})
                    save()
            reproduction.cuda_guard()
            state["status"] = "complete" if not state["diagnostic_errors"] else "measurements_complete_diagnostics_incomplete"
        except BaseException as error:
            state.update(status="failed", error=repr(error))
            raise
        finally:
            save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output, args.manifest = args.output.resolve(), args.manifest.resolve()
    if frozen.sha256(args.manifest) != frozen.MANIFEST_SHA:
        raise ValueError("Manifest identity changed")
    run(args)


if __name__ == "__main__":
    main()
