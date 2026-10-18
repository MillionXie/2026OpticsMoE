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
CONFIG = TASK / "configs/baselines/resident_busy50_groupfeedback_20261018.json"


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


def feedback_observation(samples, start, end):
    """Use whole completed work/idle groups, avoiding idle-tail aliasing."""
    values = [s.utilization_percent for s in samples if start <= s.host_monotonic_s < end]
    if len(values) < 10:
        return None
    return dict(sample_start_monotonic_s=start, sample_end_monotonic_s=end,
                sample_count=len(values), observed_percent=sum(values)/len(values))


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
    sampler = telemetry.TelemetrySampler(frozen.A100_UUID, raw_path=folder/"telemetry_stdout.jsonl")
    clocks, guard = uniform.ClockSampler(), ForeignGuard()
    groups, updates = [], []
    def safe_wait(seconds):
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            guard.check()
            sampler.check()
            if time.time() >= args.deadline:
                raise TimeoutError("Original finite deadline reached")
            time.sleep(min(0.2, max(0, until-time.monotonic())))
    def phase(name, minimum, duty, adaptive=False):
        sampler.set_phase(name); start = time.monotonic(); last_update = start
        output = None
        while time.monotonic() - start < minimum:
            guard.check()
            sampler.check()
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
                observation = feedback_observation(list(sampler.samples), last_update, end)
                if observation is not None:
                    new = update_duty(duty, observation["observed_percent"])
                    updates.append(dict(at_monotonic_s=end, **observation, old_duty=duty, new_duty=new))
                    duty = new
                    last_update = end
        return start, time.monotonic(), output, duty
    try:
        sampler.start(); clocks.start(); guard.start()
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
        persist_measurement_evidence(folder, sampler, clocks, guard, groups, updates, sys.exc_info()[1])


def persist_measurement_evidence(folder, sampler, clocks, guard, groups, updates, primary=None):
    """Retain every available raw artifact even if a reader/cleanup operation fails."""
    errors = []
    def attempt(name, action):
        try:
            return action()
        except BaseException as error:
            errors.append(dict(operation=name, error=repr(error)))
            return None
    sampler.set_phase("after")
    attempt("telemetry_stop", lambda: sampler.stop(raise_on_error=False))
    attempt("guard_stop", guard.stop)
    attempt("clock_stop", lambda: clocks.stop(folder/"clock_telemetry.csv"))
    attempt("numeric_telemetry_save", lambda: telemetry.save_samples(folder/"telemetry.csv", list(sampler.samples)))
    for name, value in [("forward_groups.json", groups), ("controller_updates.json", updates),
                        ("process_audit.json", getattr(guard, "records", []))]:
        attempt(name, lambda name=name, value=value: frozen.write_json(folder/name, value))
    attempt("telemetry_integrity", sampler.check)
    attempt("foreign_guard", guard.check)
    frozen.write_json(folder/"measurement_evidence_status.json", dict(
        status="incomplete" if primary is not None or errors else "recorded",
        primary_exception=repr(primary) if primary is not None else None,
        errors=errors, valid_numeric_samples=len(sampler.samples),
        raw_stdout="telemetry_stdout.jsonl", missing_values_never_imputed=True,
        not_a_complete_measurement_claim=True))
    if errors and primary is None:
        raise RuntimeError(f"Measurement failed; available raw evidence retained: {errors!r}")


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


def continuation_evidence(args, c):
    """Only the four verified original points may be reused; failed evidence stays intact."""
    previous = getattr(args, "resume_from", None)
    if previous is None:
        return None
    previous = previous.resolve()
    owner = TASK/"runs/simulation/lgvq_resident_busy50_a100_20261017/attempt02_20261018"
    if previous != owner.resolve() or args.output.parent.resolve() != previous.parent:
        raise ValueError("Continuation must remain in the original canonical run")
    state = json.loads((previous/"suite_status.json").read_text())
    expected = ["qwen_batch01", "qwen_batch03", "glm_batch01", "glm_batch02"]
    if state["status"] != "failed" or state["completed"] != expected or state["protocol"] != c:
        raise ValueError("Exactly four original completed points and unchanged v2 protocol required")
    bound = {p.replace("\\", "/"): h for p, h in sources().items()}
    allowed_changes = {"batch_telemetry.py", "measure_busy50_service.py"}
    hashes = {"suite_status.json": frozen.sha256(previous/"suite_status.json")}
    for name in expected:
        point = previous/name
        identity = json.loads((point/"identity.json").read_text())
        report = json.loads((point/"report.json").read_text())
        if (report["status"] != "complete" or len(report["repeats"]) != 3
                or identity["checkpoint_sha256"] != costs.CHECKPOINTS[identity["baseline"]]
                or identity["gpu_uuid"] != frozen.A100_UUID or identity["manifest_sha256"] != frozen.MANIFEST_SHA
                or identity["protocol"] != c):
            raise RuntimeError("Original completed-point identity mismatch")
        for source, old_hash in identity["source_sha256"].items():
            if Path(source).name not in allowed_changes and bound.get(source.replace("\\", "/")) != old_hash:
                raise RuntimeError(f"Unapproved arithmetic/input source change: {source}")
        required = ["identity.json", "report.json", "status.json", "measurement/telemetry.csv",
                    "measurement/clock_telemetry.csv", "measurement/forward_groups.json",
                    "measurement/controller_updates.json", "measurement/process_audit.json",
                    *[f"measurement/repeat{i:02d}.json" for i in (1, 2, 3)]]
        for relative in required:
            hashes[f"{name}/{relative}"] = frozen.sha256(point/relative)
    return dict(source=str(previous), preserved_points=expected, files_sha256=hashes,
                new_human_authorization="2026-10-18: repair and remaining8, new2h window; never rerun original4",
                telemetry_change="strict missing-row rejection plus durable raw stdout and failure cleanup; no measurement arithmetic change")


def planned_points(c, continuation=None):
    points = [(model, batch) for model, batches in c["batches"].items() for batch in batches]
    if continuation is None:
        return points
    remaining = [(model, batch) for model, batch in points
                 if f"{model}_batch{batch:02d}" not in continuation["preserved_points"]]
    if len(remaining) != 8 or remaining[0] != ("small", 1):
        raise ValueError("Exactly the authorized remaining8 points required")
    return remaining


def suite(args):
    import fcntl
    import signal
    c = config(); reproduction.require_new_output(args.output)
    continuation = continuation_evidence(args, c)
    plan = planned_points(c, continuation)
    with (TASK/"runs/simulation/.temporal_telemetry_a100.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        costs.wait_for_idle(args, c); args.output.mkdir()
        state = dict(status="running", protocol=c, pid=os.getpid(), source_sha256=sources(),
                     completed=[], deadline=args.deadline, started_unix=time.time(),
                     continuation=continuation, planned_points=[f"{m}_batch{b:02d}" for m, b in plan])
        def save():
            frozen.write_json(args.output/"suite_status.json", state)
        try:
            save()
            for baseline, batch in plan:
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
            if continuation:
                previous = Path(continuation["source"])
                if any(frozen.sha256(previous/p) != h for p, h in continuation["files_sha256"].items()):
                    raise RuntimeError("Protected original4 evidence changed")
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
    p.add_argument("--resume-from", type=Path)
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
