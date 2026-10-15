"""Sequential, dataset-once frozen LGVQ evaluation of existing batch points.

Uses the previously approved first-native-block to synchronized GPU MOS wall
boundary. Resident utilization is a separate experiment, never a cost input.
The protected Qwen B1 cost remains a historical anchor, not a new measurement.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from . import batch_telemetry as telemetry
from . import evaluate_temporal_baseline_batches as frozen
from . import reproduce_temporal_baseline as reproduction
from . import run_temporal_telemetry_suite as prior
from . import unified_resident_utilization as resident

TASK, REPO = prior.TASK, prior.REPO
CONFIG = TASK / "configs/baselines/frozen_cost_all_batches_20261015.json"
CHECKPOINTS = {
    "qwen": "e102adc6a7b33335c8f49ae1cbeea3302a1a10017d282da8d6091a640276d0d2",
    "clip": "28729f41b09c59866ed54e0d520c48edca276ac0237d0b6bd885c8a0db4619d4",
    "yolo": "47606708c66b313174d3a04edef6ebac1b06d3310d2f1049d7a989dbc386b848",
    "deepseek": "3748b08f9b4c7b61cd18425f49f899ea34598e3169846119ae75272bccdb7951",
    "glm": "666f797622e0addd3a2ac064b46c5e786c0572365bea49eed161b59c574abedc",
    "small": "c5c3e377917ffc6a78afb915e7c8f114b11654a02a328a0652a698581fdad41d",
}


def config():
    c = json.loads(CONFIG.read_text(encoding="utf-8"))
    if (c["test_videos"], c["cost_videos"], c["equivalent_videos"], c["host_proxy_w"],
            c["explicit_warmup_forwards"]) != (558, 200, 16, 338.2, 0):
        raise ValueError("Approved dataset-once cost contract changed")
    if set(c["batches"]) != set(CHECKPOINTS) or sum(map(len, c["batches"].values())) != 30:
        raise ValueError("Only the thirty existing utilization points are authorized")
    return c


def plan(batch):
    """Keep ALL samples, with explicit first-200/full-558 boundaries."""
    if not isinstance(batch, int) or batch < 1:
        raise ValueError("Positive video batch required")
    return [(start, min(start + batch, end))
            for begin, end in ((0, 200), (200, 558)) for start in range(begin, end, batch)]


def sources():
    result = resident.sources()
    for path in (Path(__file__), CONFIG, Path(reproduction.__file__),
                 REPO / "LightGenV2/common/baseline_measurement.py",
                 TASK / "visual_backbone_quality.py"):
        result[str(path.relative_to(REPO))] = frozen.sha256(path)
    return result


def cost_report(baseline, batch, measured, c):
    if (baseline, batch) == ("qwen", 1):
        anchor = c["qwen_b1_anchor"]
        t, e = anchor["latency_16_videos_ms"], anchor["energy_16_videos_j"]
        return dict(latency_16_videos_ms=t, energy_16_videos_j=e,
                    latency_per_video_ms=t/16, energy_per_video_j=e/16,
                    throughput_videos_per_second=16000/t,
                    provenance="protected_historical_Qwen_B1_anchor_not_this_round",
                    historical_srcc=anchor["historical_srcc"])
    return {**measured, "provenance": "fresh_dataset_once_same_round_power_and_wall_time"}


def worker(args, c):
    reproduction.require_new_output(args.output)
    reproduction.cuda_guard()
    if args.baseline not in resident.SPECIAL:
        frozen.preflight_files(args)
    import numpy as np
    import torch
    from . import quality_token_common as core
    from .measure_resident_utilization import ForeignGuard
    from LightGenV2.common import baseline_measurement as measurement

    rated = measurement.gpu_power_limit_w(frozen.A100_UUID)
    if abs(rated - 250) > .1:
        raise RuntimeError("A100 must retain the approved 250 W limit")
    identity = dict(baseline=args.baseline, batch_size_videos=args.batch, pid=os.getpid(),
                    gpu_uuid=frozen.A100_UUID, gpu_name=measurement.validate_cuda_device("A100"),
                    actual_power_limit_w=rated, source_sha256=sources(), command=sys.argv,
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                    manifest_sha256=frozen.sha256(args.manifest), python=platform.python_version(),
                    torch=torch.__version__, cuda=torch.version.cuda, protocol=c,
                    allocator=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
    args.output.mkdir()
    def status(state, **extra):
        frozen.write_json(args.output / "status.json", dict(status=state, **extra, **identity))
    status("loading")
    sampler, timer, handles, guard = None, None, [], ForeignGuard()
    guard.start()
    try:
        with torch.inference_mode():
            if args.baseline in resident.SPECIAL:
                model, head, prepare, transfer, compute, _, finish = resident.special_runtime(args)
                # Verified official GLM visual path and original DeepSeek vision path.
                first = model.model.visual.blocks[0] if args.baseline == "glm" else model.vision.blocks[0]
                timer = measurement.FirstBlockTimer(first)
                identity["first_block_path"] = ("model.visual.blocks.0" if args.baseline == "glm" else "vision.blocks.0")
                def forward(inputs):
                    torch.cuda.synchronize()
                    timer.reset()
                    values = compute(inputs)
                    timing = timer.finish()
                    return values.detach().cpu().tolist(), timing["host_ms"], timing["cuda_ms"]
                rows = [r for r in core.read_manifest(args.manifest) if r["split"] == "test"]
            else:
                before_assets = model_hashes(args.model)
                model, head, rows, prepare, transfer, forward, timer, handles = frozen.load_runtime(args)
                identity["model_file_sha256_before"] = before_assets
                def finish():
                    if model_hashes(args.model) != before_assets:
                        raise RuntimeError("Frozen official model files changed")
            identity.update(model=str(args.model), checkpoint=str(args.checkpoint),
                            checkpoint_sha256=frozen.sha256(args.checkpoint))
            if identity["checkpoint_sha256"] != CHECKPOINTS[args.baseline]:
                raise RuntimeError("Exact frozen checkpoint differs from authorized head")
            if len(rows) != 558 or len({r["sample_id"] for r in rows}) != 558:
                raise RuntimeError("Exactly 558 unique ordered TEST videos required")
            if any(p.requires_grad for m in (model, head) for p in m.parameters()):
                raise RuntimeError("All backbone and head parameters must remain frozen")
            status("evaluating", completed_test_videos=0)
            sampler = telemetry.TelemetrySampler(frozen.A100_UUID)
            sampler.start()
            sampler.set_phase("idle")
            time.sleep(2)
            sampler.set_phase(None)
            predictions, targets, timings = [], [], []
            with (args.output / "predictions.csv").open("x", newline="", encoding="utf-8") as ps, \
                 (args.output / "batch_timing.csv").open("x", newline="", encoding="utf-8") as ts:
                pw, tw = csv.writer(ps), csv.writer(ts)
                pw.writerow(["sample_index", "sample_id", "target_mos", "prediction", "batch_index", "selected_frame_positions"])
                tw.writerow(["batch_index", "videos", "in_cost_subset", "synchronized_wall_ms", "cuda_ms", "preprocessing_audit_ms", "h2d_audit_ms"])
                for index, (start, end) in enumerate(plan(args.batch)):
                    guard.check()
                    if time.time() >= args.deadline:
                        raise TimeoutError("Finite suite deadline reached")
                    t = time.perf_counter()
                    cpu, positions = prepare(rows[start:end])
                    prep_ms = 1000*(time.perf_counter()-t)
                    t = time.perf_counter()
                    inputs = transfer(cpu)
                    torch.cuda.synchronize()
                    h2d_ms = 1000*(time.perf_counter()-t)
                    in_cost = end <= 200
                    sampler.set_phase(f"active:{index}" if in_cost else None)
                    try:
                        values, wall, cuda = forward(inputs)
                    finally:
                        sampler.set_phase(None)
                    guard.check()
                    if len(values) != end-start or not all(math.isfinite(v) for v in values):
                        raise RuntimeError("Invalid native predictions")
                    if not (wall > 0 and cuda > 0):
                        raise RuntimeError("Missing first-native-block timing boundary")
                    timing = [index, end-start, in_cost, wall, cuda, prep_ms, h2d_ms]
                    timings.append(timing)
                    tw.writerow(timing)
                    for offset, value in enumerate(values):
                        row = rows[start+offset]
                        predictions.append(value)
                        targets.append(float(row["temporal"]))
                        pw.writerow([start+offset, row["sample_id"], row["temporal"], value, index, json.dumps(positions[offset])])
                    ps.flush(); ts.flush()
                    del inputs, cpu
                    if index == 0 or end % 50 < args.batch or end == 558:
                        status("evaluating", completed_test_videos=end)
                        print(f"[{args.baseline} B={args.batch}] {end}/558 wall={wall:.3f}ms", flush=True)
            samples = sampler.stop()
            sampler = None
            telemetry.save_samples(args.output / "telemetry.csv", samples)
            cost_rows = [r for r in timings if r[2]]
            power = measurement.power_report(samples, [r[3] for r in cost_rows], power_limit_w=rated)
            measured = telemetry.workload_cost(cost_rows, power["active_mean_w"])
            report = dict(status="complete", **identity, test_videos=558, cost_test_videos=200,
                performance=core.metrics(np.asarray(targets), np.asarray(predictions)),
                cost=cost_report(args.baseline, args.batch, measured, c),
                audit_measured_cost_not_for_Qwen_B1_replacement=measured,
                power=power, telemetry=telemetry.telemetry_report(samples), test_batch_plan=plan(args.batch),
                explicit_warmup_forwards=0, first_test_batch_included=True,
                timing_boundary="first native backbone block input to GPU MOS synchronized host wall",
                excluded_from_final_cost=["decode", "crop/resize", "processor", "H2D", "audit", "model loading"],
                energy_formula="(338.2W host proxy + same-run inference-active A100 mean W) * T16/1000",
                power_resolution_note="50ms polling, driver averaging window; tagged-window mean power, not calibrated per-call integral",
                resident_utilization="independent earlier experiment, not used in these costs",
                checkpoint_selection="existing frozen PT only; no training or TEST model selection in this run")
            status("verifying_assets", completed_test_videos=558)
            finish()
            guard.check()
            frozen.process_audit()
            if (sources() != identity["source_sha256"] or frozen.sha256(args.checkpoint) != identity["checkpoint_sha256"]
                    or frozen.sha256(args.manifest) != identity["manifest_sha256"]):
                raise RuntimeError("Bound source/checkpoint changed during evaluation")
            frozen.write_json(args.output / "report.json", report)
            status("complete", completed_test_videos=558)
            print(json.dumps(dict(baseline=args.baseline, batch=args.batch, performance=report["performance"], cost=report["cost"])), flush=True)
    except BaseException as error:
        status("failed", error=repr(error))
        raise
    finally:
        if sampler is not None:
            telemetry.save_samples(args.output / "telemetry.csv", sampler.stop())
        guard.stop()
        frozen.write_json(args.output / "process_audit.json", guard.records)
        if timer is not None:
            timer.close()
        for handle in handles:
            handle.remove()


def model_hashes(model):
    paths = sorted(p for p in model.iterdir() if p.is_file() and p.suffix in (".json", ".safetensors", ".bin")) if model.is_dir() else [model]
    return {p.name: frozen.sha256(p) for p in paths}


def child_command(baseline, batch, args, output):
    if baseline in resident.SPECIAL:
        command = [resident.SPECIAL[baseline][2], "-u", "-m", __package__ + ".measure_frozen_batch_costs",
                   "--baseline", baseline, "--batch", str(batch), "--manifest", str(args.manifest), "--output", str(output)]
    else:
        command = prior.child_command(baseline, batch, args.manifest, output)
        command[command.index(__package__ + ".evaluate_temporal_baseline_batches")] = __package__ + ".measure_frozen_batch_costs"
        command.remove("--telemetry")
    return command + ["--phase", "worker", "--deadline", str(args.deadline)]


def suite(args, c):
    import fcntl
    reproduction.require_new_output(args.output)
    with (TASK / "runs/simulation/.temporal_telemetry_a100.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for check in range(3):
            reproduction.cuda_guard()
            if check < 2:
                time.sleep(10)
        args.output.mkdir()
        bound = sources()
        state = dict(status="running", pid=os.getpid(), source_sha256=bound, protocol=c,
                     started_unix=time.time(), deadline=args.deadline, completed=[])
        def save():
            frozen.write_json(args.output / "suite_status.json", state)
        save()
        try:
            for baseline, batches in c["batches"].items():
                for batch in batches:
                    reproduction.cuda_guard()
                    if sources() != bound:
                        raise RuntimeError("Bound suite source changed")
                    if time.time() >= args.deadline:
                        raise TimeoutError("Finite suite deadline reached")
                    name = f"{baseline}_batch{batch:02d}"
                    command = child_command(baseline, batch, args, args.output / name)
                    state.update(phase=name, phase_started_unix=time.time())
                    save()
                    frozen.write_json(args.output / (name + ".command.json"), command)
                    env = os.environ.copy()
                    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
                    if baseline == "small":
                        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
                    with (args.output / (name + ".log")).open("x") as stream:
                        subprocess.run(command, cwd=REPO, env=env, check=True, stdout=stream, stderr=subprocess.STDOUT,
                                       timeout=min(c["maximum_point_seconds"], args.deadline-time.time()))
                    report = json.loads((args.output / name / "report.json").read_text())
                    if report["status"] != "complete" or report["test_videos"] != 558 or report["cost_test_videos"] != 200:
                        raise RuntimeError("Incomplete point cannot enter final table")
                    state["completed"].append(name)
                    save()
            reproduction.cuda_guard()
            state.update(status="complete", finished_unix=time.time())
        except BaseException as error:
            state.update(status="failed", error=repr(error))
            raise
        finally:
            save()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("suite", "worker"), required=True)
    p.add_argument("--baseline", choices=tuple(CHECKPOINTS))
    p.add_argument("--batch", type=int)
    for name in ("manifest", "output", "model", "checkpoint", "reproduction-receipt"):
        p.add_argument("--"+name, type=Path, required=name in ("manifest", "output"))
    p.add_argument("--deadline", type=float, required=True)
    args = p.parse_args()
    for key in ("manifest", "output", "model", "checkpoint", "reproduction_receipt"):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())
    c = config()
    if frozen.sha256(args.manifest) != frozen.MANIFEST_SHA:
        raise ValueError("Original manifest identity required")
    if not (0 < args.deadline-time.time() <= c["maximum_suite_seconds"]):
        raise ValueError("Positive finite deadline, at most eight hours required")
    if args.phase == "worker" and (args.baseline not in c["batches"] or args.batch not in c["batches"][args.baseline]):
        p.error("Only previously measured batch points are authorized")
    (suite if args.phase == "suite" else worker)(args, c)


if __name__ == "__main__":
    main()
