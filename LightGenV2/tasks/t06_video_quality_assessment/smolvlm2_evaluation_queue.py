"""CPU-only bounded handoff: certified Smol training -> fresh A100 B1/4/8.

No model computation or new measurement algorithm: invoke the published evaluator.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from . import smolvlm2_temporal_baseline as smol


def read_json(path):
    return json.loads(path.read_text())


def verify_training(path):
    receipt = read_json(path / "training_receipt.json")
    if receipt["status"] != "complete" or read_json(path / "status.json")["status"] != "complete":
        raise RuntimeError("Training not certified complete")
    if receipt["identity"]["source_sha256"] != smol.sources():
        raise RuntimeError("Training source changed; do not replace running source")
    for name, sha in receipt["artifacts"].items():
        target = (path / name).resolve()
        if not target.is_relative_to(path.resolve()) or smol.frozen.sha256(target) != sha:
            raise RuntimeError(f"Training artifact SHA mismatch: {name}")
    history = read_json(path / "train_history.json")
    if [row["epoch"] for row in history] != list(range(1, 51)):
        raise RuntimeError("Expected original 50 training epochs")
    report = read_json(path / "training_report.json")
    if report["backbone_updates"] != 0 or report["output_row_parameters"] != 10240:
        raise RuntimeError("Five-row-only training contract changed")
    cpu = read_json(path / "cpu_verification.json")
    for name in ("best", "last"):
        if not math.isfinite(cpu[name]["strict_cpu_metrics"]["srcc"]):
            raise RuntimeError("Nonfinite strictCPU score")
        if f"{name}_checkpoint.pt" not in receipt["artifacts"]:
            raise RuntimeError("Missing bound best/last checkpoint")
        with (path / f"cpu_{name}_predictions.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 558 or len({r["sample_id"] for r in rows}) != 558:
            raise RuntimeError("Expected 558 unique CPU predictions")
    return receipt


def verify_evaluation(path, batch, checkpoint_sha):
    report = read_json(path / "report.json")
    if (report["status"] != "complete" or report["batch_size_videos"] != batch
            or report["test_videos"] != 558 or report["cost_test_videos"] != 200
            or report["checkpoint_sha256"] != checkpoint_sha):
        raise RuntimeError("Incomplete or mismatched A100 result")
    with (path / "predictions.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 558 or len({r["sample_id"] for r in rows}) != 558:
        raise RuntimeError("Expected each B to produce 558 unique native predictions")
    return {"batch": batch, "checkpoint_sha256": checkpoint_sha,
            "performance": report["performance"], "cost": report["cost"],
            "telemetry": report["telemetry"]}


def idle_a100():
    smol.frozen.process_audit()
    limit = float(subprocess.check_output(["nvidia-smi", f"--id={smol.frozen.A100_UUID}",
        "--query-gpu=power.limit", "--format=csv,noheader,nounits"], text=True).strip())
    if limit != 250:
        raise RuntimeError("A100 power limit is not250; will not change it")


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    training, output = args.training.resolve(), args.output.resolve()
    smol.new_output(output)
    if training.parent != output.parent or not training.is_dir():
        raise ValueError("Queue must share the canonical training parent")
    start = read_json(training.parent / "rtx3090_training_start_20261015.json")
    source = smol.sources()
    if source != start["source_sha256"]:
        raise RuntimeError("Training source identity changed")
    total_deadline = start["created_at_unix"] + 4 * 3600
    train_deadline = start["created_at_unix"] + start["wall_time_limit_seconds"] + 30
    output.mkdir()
    state = {"pid": os.getpid(), "status": "waiting_for_training", "completed": [],
             "created_at_unix": time.time(), "total_deadline_unix": total_deadline,
             "source_sha256": source, "queue_source_sha256": smol.frozen.sha256(Path(__file__)),
             "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=smol.REPO, text=True).strip(),
             "gpu_context_while_waiting": False}
    def save():
        state["updated_at_unix"] = time.time()
        smol.frozen.write_json(output / "queue_status.json", state)
    save()
    with (smol.TASK / "runs/simulation/.smolvlm2_evaluation_queue.lock").open("a") as queue_lock:
        fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while True:
                if smol.sources() != source:
                    raise RuntimeError("Training/evaluation source changed during handoff")
                status = read_json(training / "status.json") if (training / "status.json").exists() else {}
                if status.get("status") == "failed":
                    raise RuntimeError("Training failed; preserve evidence, no automatic retry")
                if status.get("status") == "complete":
                    break
                if time.time() >= train_deadline:
                    raise RuntimeError("Original two-hour training bound reached")
                state["training_progress"] = {k: status[k] for k in ("status", "videos", "epoch") if k in status}
                save()
                time.sleep(15)
            receipt = verify_training(training)
            checkpoint = training / "best_checkpoint.pt"
            checkpoint_sha = smol.frozen.sha256(checkpoint)
            state.update(status="waiting_for_idle_a100", checkpoint_sha256=checkpoint_sha,
                         training_receipt_sha256=smol.frozen.sha256(training / "training_receipt.json"),
                         strict_cpu_development=read_json(training / "cpu_verification.json"))
            wait_deadline = min(time.time() + 3600, total_deadline)
            state["a100_wait_deadline_unix"] = wait_deadline
            with (smol.TASK / "runs/simulation/.temporal_telemetry_a100.lock").open("a") as gpu_lock:
                while True:
                    if smol.sources() != source:
                        raise RuntimeError("Evaluation source changed while CPU waiting")
                    if time.time() >= wait_deadline:
                        raise RuntimeError("Bounded A100 wait expired; no GPU context created")
                    acquired = False
                    try:
                        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        acquired = True
                        idle_a100()
                        break
                    except BlockingIOError:
                        state["waiting_reason"] = "shared A100 lock held by another task"
                    except RuntimeError as error:
                        if "occupied by other compute processes" not in str(error):
                            raise
                        state["waiting_reason"] = str(error)
                    if acquired:
                        fcntl.flock(gpu_lock, fcntl.LOCK_UN)
                    save()
                    time.sleep(15)
                state["a100_acquired_at_unix"] = time.time()
                for batch in (1, 4, 8):
                    if smol.sources() != source:
                        raise RuntimeError("Source changed; no new measurement dispatched")
                    idle_a100()
                    target = training.parent / f"a100_batch{batch:02d}_20261015"
                    if target.exists():
                        raise FileExistsError("Existing A100 result protected; no duplicate")
                    command = [sys.executable, "-u", "-m", smol.__name__, "--phase", "evaluate",
                        "--model", receipt["identity"]["original_command"][receipt["identity"]["original_command"].index("--model")+1],
                        "--manifest", str(smol.REPO / "experiments/lgvq_four_stage_optical_electronic_109_no_attention_vqa/artifacts/lgvq_train2250_test558.csv"),
                        "--checkpoint", str(checkpoint), "--batch", str(batch), "--output", str(target)]
                    smol.frozen.write_json(output / f"batch{batch:02d}.command.json", command)
                    state.update(status="evaluating", batch=batch)
                    save()
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=smol.frozen.A100_UUID, PYTHONUNBUFFERED="1")
                    remaining = min(1800, total_deadline-time.time())
                    if remaining <= 0:
                        raise RuntimeError("Total handoff bound reached")
                    with (output / f"batch{batch:02d}.log").open("x") as log:
                        subprocess.run(command, cwd=smol.REPO, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, check=True, timeout=remaining)
                    result = verify_evaluation(target, batch, checkpoint_sha)
                    state["completed"].append(result)
                    save()
                state.update(status="complete", a100_release_audit=smol.frozen.process_audit())
        except BaseException as error:
            state.update(status="failed", error=repr(error))
            raise
        finally:
            save()


if __name__ == "__main__":
    main()
