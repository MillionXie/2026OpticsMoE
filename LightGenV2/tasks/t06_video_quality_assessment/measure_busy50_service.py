"""Supplementary time-busy throttling, NOT optical spatial-occupancy matching.

Frozen resident input, calibrated pacing, three complete fixed windows. All
idle time and same-window power enter service cost. No paper anchors replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from . import batch_telemetry as telemetry
from . import evaluate_temporal_baseline_batches as frozen
from . import measure_frozen_batch_costs as costs
from . import measure_selected_execution as selected
from . import reproduce_temporal_baseline as reproduction
from . import unified_resident_utilization as uniform
from .measure_resident_utilization import ForeignGuard, summarize

TASK, REPO = uniform.TASK, uniform.REPO
CONFIG = TASK / "configs/baselines/resident_busy50_20261017.json"


def config():
    c = json.loads(CONFIG.read_text(encoding="utf-8"))
    expected = {"qwen": [1, 3], "glm": [1, 2], "small": [1, 3],
                "clip": [1, 10], "yolo": [1, 3], "deepseek": [1, 8]}
    if c["batches"] != expected or (c["target_gpu_busy_percent"], c["repeats"],
            c["queue_depth"], c["measurement_seconds"], c["host_proxy_w"]) != (50, 3, 8, 30, 338.2):
        raise ValueError("Fixed supplementary protocol changed")
    return c


def update_duty(duty, observed, target=50.0):
    """Only slow real inference; never add unrelated work to raise a low rate."""
    if not all(math.isfinite(x) for x in (duty, observed, target)) or not 0 <= observed <= 100:
        raise ValueError("Invalid controller observation")
    ratio = min(1.5, max(0.67, target / max(observed, 1.0)))
    return min(1.0, max(0.05, duty * ratio))


def imposed_wait(compute_seconds, duty):
    if compute_seconds <= 0 or not 0.05 <= duty <= 1:
        raise ValueError("Positive real-forward time and bounded duty required")
    return compute_seconds * (1 / duty - 1)


def service_cost(duration_s, calls, batch, board_w, host_w=338.2):
    if min(duration_s, calls, batch, board_w) <= 0:
        raise ValueError("Complete window, calls and measured power required")
    videos = calls * batch
    t16 = duration_s * 1000 * 16 / videos
    return dict(completed_video_evaluations=videos, service_time_16_videos_ms=t16,
                estimated_energy_16_videos_j=(host_w + board_w) * t16 / 1000,
                measured_same_window_board_w=board_w, host_proxy_w=host_w,
                includes_all_imposed_wait=True,
                not_dataset_once_latency=True, unique_videos=batch)


def sources():
    return {**selected.sources(), **{p.relative_to(REPO).as_posix(): frozen.sha256(p)
                                    for p in (Path(__file__), CONFIG)}}


def load(args):
    """Reuse already published arithmetic; no process patch or runtime wrapper."""
    import torch
    from . import quality_token_common as core
    if args.baseline == "qwen":
        model, head, rows, prepare, transfer, native, build, finish, _ = selected.qwen_runtime(args)
        cpu, positions = prepare(rows[:args.batch]); inputs = transfer(cpu)
        reference = native(inputs).detach().cpu()
        adapter = None
        if args.batch == 1:
            compute = native
        else:
            adapter, compute = build(cpu, inputs)
        # Keep the adapter alive via the compute closure and explicit return.
        retained = adapter
    elif args.baseline in uniform.SPECIAL:
        model, head, prepare, transfer, compute, reference_fn, finish = uniform.special_runtime(args)
        rows = [r for r in core.read_manifest(args.manifest) if r["split"] == "test"]
        cpu, positions = prepare(rows[:args.batch]); inputs = transfer(cpu)
        reference = reference_fn(inputs).detach().cpu(); retained = None
    else:
        before = costs.model_hashes(args.model)
        model, head, rows, prepare, transfer, native, timer, handles = frozen.load_runtime(args)
        cpu, positions = prepare(rows[:args.batch]); inputs = transfer(cpu)
        reference = torch.tensor(native(inputs)[0]); timer.close()
        compute = uniform.execution.make_gpu_forward(args, model, head, native)
        def finish():
            if costs.model_hashes(args.model) != before:
                raise RuntimeError("Official frozen model assets changed")
            for handle in handles:
                handle.remove()
        retained = None
    if len(rows) != 558 or any(p.requires_grad for m in (model, head) for p in m.parameters()):
        raise RuntimeError("Original558 TEST and frozen weights required")
    if frozen.sha256(args.checkpoint) != costs.CHECKPOINTS[args.baseline]:
        raise RuntimeError("Exact frozen head identity required")
    if not isinstance(cpu, dict):
        fields = {"frames": cpu} if torch.is_tensor(cpu) else vars(cpu)
    else:
        fields = cpu
    fingerprints = {k: dict(shape=list(v.shape), dtype=str(v.dtype), sha256=hashlib.sha256(
        v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
        for k, v in fields.items() if torch.is_tensor(v)}
    if not fingerprints:
        raise RuntimeError("Missing input tensor identity")
    result = compute(inputs).detach().cpu()
    if not torch.isfinite(result).all() or not torch.allclose(result, reference, atol=0.0002, rtol=0):
        raise RuntimeError("Native prediction parity failed")
    return compute, inputs, reference, finish, dict(sample_ids=[r["sample_id"] for r in rows[:args.batch]],
        frame_positions=positions, input_tensors=fingerprints,
        native_reference_mos=reference.tolist(), initial_mos=result.tolist()), retained


def measure(args, compute, inputs, folder, c):
    import torch
    folder.mkdir()
    sampler, clocks, guard = telemetry.TelemetrySampler(frozen.A100_UUID), uniform.ClockSampler(), ForeignGuard()
    groups, updates = [], []
    sampler.start(); clocks.start(); guard.start()
    def safe_wait(seconds):
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            guard.check()
            if time.time() >= args.deadline:
                raise TimeoutError("Original finite deadline reached")
            time.sleep(min(0.2, max(0, until-time.monotonic())))
    def phase(name, minimum, duty, adaptive=False):
        sampler.set_phase(name); start = time.monotonic(); last_update = start
        output = None
        while time.monotonic() - start < minimum:
            guard.check()
            if time.time() >= args.deadline:
                raise TimeoutError("Original finite deadline reached")
            tick = time.monotonic()
            for _ in range(c["queue_depth"]):
                output = compute(inputs)
            torch.cuda.synchronize(); computed = time.monotonic()
            delay = imposed_wait(computed-tick, duty)
            safe_wait(delay)
            end = time.monotonic()
            groups.append(dict(phase=name, start_monotonic_s=tick, compute_end_monotonic_s=computed,
                end_monotonic_s=end, requested_idle_s=delay, duty=duty, forwards=c["queue_depth"]))
            if adaptive and end-last_update >= c["feedback_interval_seconds"]:
                recent = [s.utilization_percent for s in list(sampler.samples)
                          if max(start, end-c["feedback_sample_window_seconds"]) <= s.host_monotonic_s < end]
                if len(recent) >= 10:
                    observed = sum(recent)/len(recent); new = update_duty(duty, observed)
                    updates.append(dict(at_monotonic_s=end, observed_percent=observed, old_duty=duty, new_duty=new))
                    duty = new
                last_update = end
        return start, time.monotonic(), output, duty
    try:
        _, _, _, duty = phase("calibration", c["calibration_seconds"], c["initial_duty_fraction"], True)
        repeats = []
        for repeat in range(1, c["repeats"]+1):
            phase(f"warmup{repeat}", c["warmup_seconds"], duty)
            cpu_start = uniform.cpu_snapshot()
            start, end, output, _ = phase(f"measured{repeat}", c["measurement_seconds"], duty)
            cpu_end = uniform.cpu_snapshot()
            summary = summarize(list(sampler.samples), start, end)
            count = sum(g["forwards"] for g in groups if g["phase"] == f"measured{repeat}")
            repeats.append(dict(repeat=repeat, measurement=summary, frozen_duty=duty,
                forward_calls=count, output=output.detach().cpu().tolist(),
                cpu_start=cpu_start, cpu_end=cpu_end,
                cost=service_cost(end-start, count, args.batch, summary["mean_board_power_w"])))
            frozen.write_json(folder/f"repeat{repeat:02d}.json", repeats[-1])
        guard.check()
        pooled = uniform.pooled_mean(repeats)
        duration = sum(r["measurement"]["duration_s"] for r in repeats)
        mean_w = sum(r["measurement"]["mean_board_power_w"]*r["measurement"]["duration_s"] for r in repeats)/duration
        result = dict(status="complete", calibration_updates=updates, frozen_duty=duty, repeats=repeats,
            mean_gpu_busy_percent=pooled, target_reached=abs(pooled-50) <= c["target_tolerance_pp"],
            repeat_means_percent=[r["measurement"]["mean_gpu_utilization_percent"] for r in repeats],
            pooled_service_cost=service_cost(duration, sum(r["forward_calls"] for r in repeats), args.batch, mean_w),
            accuracy="No new full558 evaluation: firstB repeated outputs only", no_window_or_repeat_selection=True)
        return result
    finally:
        sampler.set_phase("after"); samples = sampler.stop(); guard.stop()
        clocks.stop(folder/"clock_telemetry.csv")
        telemetry.save_samples(folder/"telemetry.csv", samples)
        frozen.write_json(folder/"forward_groups.json", groups)
        frozen.write_json(folder/"controller_updates.json", updates)
        frozen.write_json(folder/"process_audit.json", guard.records)
        guard.check()


def worker(args):
    import torch
    from LightGenV2.common.baseline_measurement import gpu_power_limit_w
    c = config(); reproduction.require_new_output(args.output); reproduction.cuda_guard()
    if args.batch not in c["batches"][args.baseline]:
        raise ValueError("Only predeclared12 supplementary points")
    if args.baseline not in uniform.SPECIAL:
        frozen.preflight_files(args)
    if abs(gpu_power_limit_w(frozen.A100_UUID)-250) > .1:
        raise RuntimeError("Unchanged250W limit required")
    args.output.mkdir(); finish = None; bound = sources(); guard = ForeignGuard(); guard.start()
    frozen.write_json(args.output/"status.json", dict(status="loading", pid=os.getpid(), updated_unix=time.time()))
    try:
        with torch.inference_mode():
            compute, inputs, reference, finish, identities, retained = load(args)
            frozen.write_json(args.output/"identity.json", dict(protocol=c, source_sha256=bound,
                baseline=args.baseline, batch=args.batch, gpu_uuid=frozen.A100_UUID, pid=os.getpid(),
                checkpoint_sha256=frozen.sha256(args.checkpoint), manifest_sha256=frozen.sha256(args.manifest),
                command=sys.argv, git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                **identities))
            guard.check()
            frozen.write_json(args.output/"status.json", dict(status="measuring", pid=os.getpid(), updated_unix=time.time()))
            report = measure(args, compute, inputs, args.output/"measurement", c)
            for r in report["repeats"]:
                values = torch.tensor(r["output"])
                if not torch.allclose(values, reference, rtol=0, atol=.0002):
                    raise RuntimeError("Frozen output changed during throttling")
            finish(); finish = None; guard.check()
            if sources() != bound or frozen.sha256(args.checkpoint) != costs.CHECKPOINTS[args.baseline]:
                raise RuntimeError("Source or checkpoint changed")
            frozen.write_json(args.output/"report.json", report)
            frozen.write_json(args.output/"status.json", dict(status="complete", pid=os.getpid(), updated_unix=time.time()))
            print(json.dumps(dict(baseline=args.baseline, batch=args.batch,
                utilization=report["mean_gpu_busy_percent"], target_reached=report["target_reached"])), flush=True)
    finally:
        guard.stop(); frozen.write_json(args.output/"loading_process_audit.json", guard.records)
        if finish is not None:
            finish()


def suite(args):
    import fcntl
    import signal
    c = config(); reproduction.require_new_output(args.output)
    with (TASK/"runs/simulation/.temporal_telemetry_a100.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        costs.wait_for_idle(args, c); args.output.mkdir()
        state = dict(status="running", protocol=c, pid=os.getpid(), source_sha256=sources(),
                     completed=[], deadline=args.deadline, started_unix=time.time())
        def save():
            frozen.write_json(args.output/"suite_status.json", state)
        try:
            save()
            for baseline, batches in c["batches"].items():
                for batch in batches:
                    reproduction.cuda_guard()
                    if time.time() >= args.deadline or sources() != state["source_sha256"]:
                        raise RuntimeError("Finite deadline or source changed")
                    point = f"{baseline}_batch{batch:02d}"; state["phase"] = point; save()
                    command = costs.child_command(baseline, batch, args, args.output/point)
                    command[command.index(__package__+".measure_frozen_batch_costs")] = __package__+".measure_busy50_service"
                    frozen.write_json(args.output/(point+".command.json"), command)
                    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
                    if baseline == "small":
                        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
                    with (args.output/(point+".log")).open("x") as log:
                        child = subprocess.Popen(command, cwd=REPO, env=env, stdout=log,
                                                 stderr=subprocess.STDOUT, start_new_session=True)
                        try:
                            code = child.wait(timeout=min(c["maximum_stage_seconds"], max(1, args.deadline-time.time())))
                        except BaseException:
                            os.killpg(child.pid, signal.SIGTERM)
                            try:
                                child.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=10)
                            raise
                    if code:
                        raise RuntimeError(f"{point} failed with exit{code}; no automatic retry")
                    report = json.loads((args.output/point/"report.json").read_text())
                    if report["status"] != "complete":
                        raise RuntimeError("Incomplete point")
                    state["completed"].append(point); save()
            reproduction.cuda_guard(); state.update(status="complete", finished_unix=time.time())
        except BaseException as error:
            state.update(status="failed", error=repr(error)); raise
        finally:
            save()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["suite", "worker"], required=True)
    p.add_argument("--baseline", choices=list(config()["batches"]))
    p.add_argument("--batch", type=int)
    for name in ("manifest", "output", "model", "checkpoint", "reproduction-receipt"):
        p.add_argument("--"+name, type=Path, required=name in ("manifest", "output"))
    p.add_argument("--deadline", type=float, required=True)
    args = p.parse_args(); args.output = args.output.resolve()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != frozen.A100_UUID or frozen.sha256(args.manifest) != frozen.MANIFEST_SHA:
        raise RuntimeError("Original A100 and split required")
    if not 0 < args.deadline-time.time() <= config()["maximum_suite_seconds"]:
        raise ValueError("At most3h including at most1h CPU-only idle waiting")
    try:
        (suite if args.phase == "suite" else worker)(args)
    except BaseException as error:
        failure = args.output/"failure.json" if args.output.exists() else args.output.with_suffix(".failure.json")
        frozen.write_json(failure, dict(error=repr(error), pid=os.getpid(), at_unix=time.time(), no_automatic_retry=True))
        raise


if __name__ == "__main__":
    main()
