"""Bounded utilization execution controls; historical paper data are read-only.

Three predeclared 30-second controls: original synchronized CPU-output loop,
same boundary with GPU output, and eight-forward GPU-output queue. The queue
reports full-forward throughput, NOT the historical first-block host latency.
No precision, model, input, checkpoint, or quality-head training changes.
"""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

from . import evaluate_temporal_baseline_batches as frozen
from . import run_temporal_telemetry_suite as prior
from . import batch_telemetry as telemetry

TASK = Path(__file__).resolve().parent
REPO = TASK.parents[2]
OLD = TASK / "runs/simulation/lgvq_temporal_telemetry_a100_20261014/full_remeasurement02"
CASES = (("qwen", 1), ("qwen", 8), ("clip", 8), ("clip", 10),
         ("yolo", 3), ("yolo", 8), ("deepseek", 8))
MODES = ("legacy_sync_cpu", "sync_gpu_output", "queued_gpu_output")
DURATION_S = 30
STEADY_START_S = 5
QUEUE_DEPTH = 8


def moments(values):
    values = list(values)
    if not values:
        return None
    return dict(n=len(values), mean=statistics.mean(values), min=min(values), max=max(values))


def window_summary(samples, started, duration, start=0):
    selected = [s for s in samples if started + start <= s.host_monotonic_s < started + duration]
    return dict(start_s=start, stop_s=duration, samples=len(selected),
                utilization_percent=moments(s.utilization_percent for s in selected),
                board_power_w=moments(s.watts for s in selected))


def historical_audit(output):
    """Fixed windows, no best-window selection or source file writes."""
    results, hashes = [], {}
    for folder in sorted(OLD.glob("*_batch*")):
        if not folder.is_dir() or not re.fullmatch(r"(qwen|clip|yolo|deepseek)_batch\d{2}", folder.name):
            continue
        source = folder / "sustained_telemetry.csv"
        rows = list(csv.DictReader(source.open()))
        active = [r for r in rows if r["phase"].startswith("sustained:")]
        first = float(active[0]["host_monotonic_s"])
        windows = {}
        windows["all_active"] = moments(float(r["utilization_percent"]) for r in active)
        for begin, end in ((0, 1), (0, 2), (0, 5), (0, 15), (1, 15), (2, 15), (5, 15)):
            values = [float(r["utilization_percent"]) for r in active
                      if begin <= float(r["host_monotonic_s"]) - first < end]
            windows[f"{begin}_{end}s"] = moments(values)
        calls = list(csv.DictReader((folder / "sustained_timing.csv").open()))
        timing = list(csv.DictReader((folder / "batch_timing.csv").open()))
        cost = [r for r in timing if r["in_cost_subset"] == "True"]
        results.append(dict(case=folder.name, windows=windows,
            first_resident_call_ms=float(calls[0]["synchronized_wall_ms"]),
            later_resident_calls_ms=moments(float(r["synchronized_wall_ms"]) for r in calls[1:]),
            summed_cost_boundary_ms=sum(float(r["synchronized_wall_ms"]) for r in cost),
            summed_cost_preparation_ms=sum(float(r["preprocessing_audit_ms"]) for r in cost),
            summed_cost_h2d_ms=sum(float(r["h2d_audit_ms"]) for r in cost)))
        for name in ("sustained_telemetry.csv", "sustained_timing.csv", "batch_timing.csv", "report.json"):
            hashes[f"{folder.name}/{name}"] = frozen.sha256(folder / name)
    frozen.write_json(output / "historical_windows.json", dict(rows=results, source_sha256=hashes,
        note="Elapsed windows referenced to first recorded active sample, not exact phase transition. Fixed exploratory windows, not replacement formal utilization."))


def make_gpu_forward(args, model, head, original_forward):
    """Reuse existing feature closure; only Qwen's small readout is explicit."""
    import torch
    scores = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["level_scores"].float().to("cuda:0")
    if args.baseline != "qwen":
        features = inspect.getclosurevars(original_forward).nonlocals["features"]

        def compute(inputs):
            return (features(inputs) @ head.weight.t()).softmax(-1) @ scores
    else:
        def compute(inputs):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = model.model(**inputs, return_dict=True, use_cache=False).last_hidden_state
            mask = inputs["attention_mask"].bool()
            indices = torch.arange(mask.shape[1], device=hidden.device).expand_as(mask)
            last = indices.masked_fill(~mask, -1).amax(1)
            pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last].float()
            # Preserve the full native vocabulary projection in the original work.
            native_full_logits = model.lm_head(pooled.to(model.lm_head.weight.dtype))
            prediction = head(pooled).softmax(-1) @ scores
            return prediction
    return compute


def boundary_forward(compute, timer, inputs, baseline):
    import torch
    torch.cuda.synchronize()
    timer.reset()
    prediction = compute(inputs)
    data = timer.finish()
    key = "vision_first_block_to_score_host_ms" if baseline == "qwen" else "host_ms"
    return prediction, data[key]


def measure_mode(args, mode, forward, compute, timer, resident):
    import torch
    sampler = telemetry.TelemetrySampler(frozen.A100_UUID)
    sampler.start()
    sampler.set_phase("settling")
    time.sleep(2)
    frozen.process_audit()
    torch.cuda.synchronize()
    calls, last_value = [], None
    sampler.set_phase(mode)
    started = time.monotonic()
    try:
        while time.monotonic() - started < DURATION_S:
            tick = time.monotonic()
            if mode == "legacy_sync_cpu":
                values, boundary_ms, _ = forward(resident)
                last_value = values
                count = 1
            elif mode == "sync_gpu_output":
                last_value, boundary_ms = boundary_forward(compute, timer, resident, args.baseline)
                count = 1
            else:
                for _ in range(QUEUE_DEPTH):
                    last_value = compute(resident)
                torch.cuda.synchronize()
                boundary_ms, count = None, QUEUE_DEPTH
            calls.append(dict(start_s=tick-started, end_s=time.monotonic()-started,
                              calls=count, boundary_wall_ms=boundary_ms))
        torch.cuda.synchronize()
        ended = time.monotonic()
        sampler.set_phase("after_measurement")
    finally:
        samples = sampler.stop()
        telemetry.save_samples(args.output / f"{mode}_telemetry.csv", samples)
    frozen.write_json(args.output / f"{mode}_calls.json", calls)
    if torch.is_tensor(last_value):
        last_value = last_value.detach().cpu().tolist()
    total = sum(r["calls"] for r in calls)
    windows = [window_summary(samples, started, DURATION_S),
               window_summary(samples, started, DURATION_S, STEADY_START_S)]
    # Paired steady throughput counts only complete groups inside the fixed window.
    steady = [r for r in calls if r["start_s"] >= STEADY_START_S and r["end_s"] <= DURATION_S]
    steady_duration = sum(r["end_s"]-r["start_s"] for r in steady)
    result = dict(mode=mode, calls=total, duration_s=ended-started, windows=windows,
        complete_loop_time16_ms=(ended-started)*16000/(total*args.batch),
        steady_complete_groups=len(steady),
        steady_complete_loop_time16_ms=(steady_duration*16000/(sum(r["calls"] for r in steady)*args.batch)
                                      if steady else None),
        boundary_wall_ms=moments(r["boundary_wall_ms"] for r in calls if r["boundary_wall_ms"] is not None),
        last_mos=last_value, process_audit=frozen.process_audit(),
        timing_note="Queue costs are full-forward throughput with groups of eight, NOT historical first-block synchronous host latency. No old cost replacement.")
    if windows[0]["board_power_w"]:
        windows[0]["energy16_using_full_loop_j"] = (338.2 + windows[0]["board_power_w"]["mean"]) * result["complete_loop_time16_ms"] / 1000
    if windows[1]["board_power_w"] and result["steady_complete_loop_time16_ms"]:
        windows[1]["energy16_using_steady_groups_j"] = (338.2 + windows[1]["board_power_w"]["mean"]) * result["steady_complete_loop_time16_ms"] / 1000
    return result


def run_gpu(args):
    import torch
    from . import reproduce_temporal_baseline as reproduction
    if os.environ.get("CUDA_VISIBLE_DEVICES") != frozen.A100_UUID:
        raise RuntimeError("Wrong GPU UUID")
    reproduction.cuda_guard()
    frozen.preflight_files(args)
    args.output.mkdir(parents=True, exist_ok=False)
    identity = dict(status="loading", pid=os.getpid(), baseline=args.baseline, batch=args.batch,
        gpu_uuid=frozen.A100_UUID, checkpoint_sha256=frozen.sha256(args.checkpoint),
        manifest_sha256=frozen.sha256(args.manifest),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256={p.name: frozen.sha256(p) for p in (Path(__file__), Path(frozen.__file__), Path(telemetry.__file__))},
        duration_s=DURATION_S, fixed_steady_window_s=[STEADY_START_S, DURATION_S],
        historical_qwen_b1_anchor=dict(time16_ms=1188.3861464262009, energy16_j=462.1134010767563,
                                       measured_utilization=None, unchanged=True))
    frozen.write_json(args.output / "status.json", identity)
    timer, handles = None, []
    try:
        with torch.inference_mode():
            model, head, rows, prepare, transfer, forward, timer, handles = frozen.load_runtime(args)
            if len(rows) != 558 or any(p.requires_grad for m in (model, head) for p in m.parameters()):
                raise RuntimeError("Frozen full TEST identity required")
            # A fresh, same-order Qwen B1 first-200 audit. Never forces the anchor.
            if args.baseline == "qwen" and args.batch == 1:
                sampler = telemetry.TelemetrySampler(frozen.A100_UUID)
                sampler.start()
                timings = []
                try:
                    for i, row in enumerate(rows[:200]):
                        if i % 25 == 0:
                            frozen.process_audit()
                        tick = time.monotonic()
                        cpu, _ = prepare([row])
                        prepared = time.monotonic()
                        inputs = transfer(cpu)
                        torch.cuda.synchronize()
                        transferred = time.monotonic()
                        sampler.set_phase(f"active:{i}")
                        values, wall, cuda = forward(inputs)
                        sampler.set_phase(None)
                        timings.append([i, 1, True, wall, cuda, (prepared-tick)*1000, (transferred-prepared)*1000])
                        if i % 25 == 0:
                            frozen.write_json(args.output / "status.json", {**identity, "status": "b1_anchor_audit", "videos": i+1})
                    torch.cuda.synchronize()
                finally:
                    samples = sampler.stop()
                    telemetry.save_samples(args.output / "b1_first200_telemetry.csv", samples)
                with (args.output / "b1_first200_timing.csv").open("x", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["index", "videos", "cost", "wall_ms", "cuda_ms", "prepare_ms", "h2d_ms"])
                    writer.writerows(timings)
                active = telemetry.telemetry_report(samples)
                frozen.write_json(args.output / "b1_new_audit_only.json", dict(telemetry=active,
                    cost=telemetry.workload_cost(timings, active["mean_board_power_w"]),
                    replaces_historical_anchor=False))
            from .diagnose_batch_numerics import input_records
            cpu, positions = prepare(rows[:args.batch])
            fingerprints = input_records(cpu, args.batch)
            resident = transfer(cpu)
            torch.cuda.synchronize()
            compute = make_gpu_forward(args, model, head, forward)
            report = {**identity, "status": "running", "sample_ids": [r["sample_id"] for r in rows[:args.batch]],
                      "frame_positions": positions, "input_fingerprints": fingerprints, "modes": [],
                      "preceding_forwards": 200 if args.baseline == "qwen" and args.batch == 1 else 0,
                      "order_limitation": "Fixed legacy then GPU-output then queue; separate cold-inclusive and fixed steady windows, not randomized causal effect size."}
            # Original CPU-output mode runs first, retaining its timer hooks.
            for mode in MODES:
                if mode == "queued_gpu_output":
                    timer.close()
                frozen.write_json(args.output / "status.json", {**identity, "status": "measuring", "mode": mode})
                result = measure_mode(args, mode, forward, compute, timer, resident)
                if report["modes"]:
                    ref = report["modes"][0]["last_mos"]
                    delta = max(abs(a-b) for a, b in zip(ref, result["last_mos"]))
                    result["max_abs_mos_vs_legacy"] = delta
                    if delta > 0.0002:
                        raise RuntimeError(f"Changed GPU-output work: {delta}")
                report["modes"].append(result)
                frozen.write_json(args.output / "report.json", report)
            report["status"] = "complete"
            frozen.write_json(args.output / "report.json", report)
            frozen.write_json(args.output / "status.json", {**identity, "status": "complete"})
    except BaseException as error:
        frozen.write_json(args.output / "status.json", {**identity, "status": "failed", "error": repr(error)})
        raise
    finally:
        if timer is not None:
            timer.close()
        for handle in handles:
            handle.remove()


def suite(args):
    import fcntl
    from . import reproduce_temporal_baseline as reproduction
    reproduction.require_new_output(args.output)
    args.output.mkdir(parents=True, exist_ok=False)
    source_sha = frozen.sha256(Path(__file__))
    state = dict(status="waiting", pid=os.getpid(), deadline_unix=args.deadline, completed=[],
                 source_sha256=source_sha,
                 source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                 cases=CASES, no_cuda_while_waiting=True)
    with (TASK / "runs/simulation/.temporal_telemetry_a100.lock").open("a") as lock:
        while time.time() < args.deadline:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                reproduction.cuda_guard()
                break
            except (BlockingIOError, RuntimeError) as error:
                fcntl.flock(lock, fcntl.LOCK_UN)
                state["last_wait_reason"] = str(error)
                state["checked_unix"] = time.time()
                frozen.write_json(args.output / "suite_status.json", state)
                time.sleep(15)
        else:
            state["status"] = "deadline_wait_expired"
            frozen.write_json(args.output / "suite_status.json", state)
            return
        state["status"] = "running"
        for baseline, batch in CASES:
            if frozen.sha256(Path(__file__)) != source_sha:
                raise RuntimeError("Execution-control source changed while queued")
            remaining = args.deadline-time.time()
            if remaining < 240:
                state["status"] = "deadline_partial"
                break
            reproduction.cuda_guard()
            command = prior.child_command(baseline, batch, args.manifest,
                                           args.output / f"{baseline}_batch{batch:02d}")
            command[command.index(prior.__package__+".evaluate_temporal_baseline_batches")] = __package__+".audit_utilization_execution"
            command.remove("--telemetry")
            command.extend(["--phase", "gpu"])
            name = f"{baseline}_batch{batch:02d}"
            state["phase"] = name
            frozen.write_json(args.output / "suite_status.json", state)
            frozen.write_json(args.output / f"{name}.command.json", command)
            try:
                with (args.output / f"{name}.log").open("x") as stream:
                    subprocess.run(command, cwd=REPO, check=True, stdout=stream, stderr=subprocess.STDOUT,
                                   timeout=min(600, remaining))
                state["completed"].append(name)
            except BaseException as error:
                state.update(status="failed", error=repr(error))
                frozen.write_json(args.output / "suite_status.json", state)
                raise
        else:
            state["status"] = "complete"
        frozen.write_json(args.output / "suite_status.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("cpu", "gpu", "suite"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--baseline", choices=tuple(frozen.CHECKPOINT_SHA))
    parser.add_argument("--batch", type=int, choices=(1, 3, 8, 10))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--reproduction-receipt", type=Path)
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    from . import reproduce_temporal_baseline as reproduction
    reproduction.require_new_output(args.output)
    if args.phase == "cpu":
        args.output.mkdir(parents=True, exist_ok=False)
        historical_audit(args.output)
    elif args.phase == "gpu":
        run_gpu(args)
    else:
        if args.deadline is None or not 0 < args.deadline-time.time() <= 7200:
            raise ValueError("Explicit deadline within two hours required")
        suite(args)


if __name__ == "__main__":
    main()
