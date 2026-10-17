"""Rental-only frozen same-process batch ascent and separate paced service.

No graph, backend, precision, training, power or clock changes. All raw windows
are retained. Model loading occurs once per model, never once per batch.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from . import rental_vlm_batch_followup as previous
from . import rental_vlm_temporal as old
from . import batch_telemetry as telemetry

CONFIG = old.TASK / "configs/baselines/rental_sequential_busy50_20261018.json"
ROOT = old.RUN / "sequential_busy50_20261018"


def config():
    value = json.loads(CONFIG.read_text())
    if (value["queue_depth"], value["repeats"], value["measurement_seconds"],
            value["target_gpu_busy_percent"], value["power_limit_w"]) != (8, 3, 30, 50, 400):
        raise ValueError("Fixed rental native protocol changed")
    return value


def sources():
    return {**previous.sources(), **{p.relative_to(old.REPO).as_posix(): old.sha(p)
                                     for p in (Path(__file__), CONFIG)}}


class Guard(previous.Guard):
    def check(self):
        if time.time() >= self.args.deadline or sources() != self.identity["sources"]:
            raise RuntimeError("Finite deadline or exact bound source changed")
        raw = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
                                       "--format=csv,noheader,nounits"], text=True)
        rows = [r for r in csv.reader(raw.splitlines()) if len(r) == 3 and r[1].strip() == self.args.gpu_uuid]
        self.audit.append(dict(time=time.time(), processes=rows))
        if {r[0].strip() for r in rows} != (self.expected or set()):
            raise RuntimeError("Exclusive GPU process identity changed")


def update_duty(duty, observed):
    if not all(math.isfinite(v) for v in (duty, observed)) or not 0 <= observed <= 100:
        raise ValueError("Invalid driver observation")
    return min(1., max(.05, duty * min(1.5, max(.67, 50 / max(observed, 1.)))))


def imposed_wait(computed, duty):
    if computed <= 0 or not .05 <= duty <= 1:
        raise ValueError("Positive actual compute and bounded duty required")
    return computed * (1 / duty - 1)


def service_cost(seconds, calls, batch, board_w):
    if min(seconds, calls, batch, board_w) <= 0:
        raise ValueError("Complete real service window required")
    t16 = seconds * 1000 * 16 / (calls * batch)
    return dict(service_time_16_videos_ms=t16,
                estimated_energy_16_videos_j=(338.2 + board_w) * t16 / 1000,
                measured_same_window_board_w=board_w, host_proxy_w=338.2,
                completed_video_evaluations=calls * batch, unique_videos=batch,
                includes_all_imposed_wait=True, not_dataset_once_latency=True)


def reached_target(report):
    return report["pooled_mean_utilization_percent"] > 98


def input_identity(cpu, selected, positions):
    import torch
    return dict(sample_ids=[r["sample_id"] for r in selected], frame_positions=positions,
        tensors={k: dict(shape=list(v.shape), dtype=str(v.dtype), sha256=hashlib.sha256(
            v.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
            for k, v in cpu.items() if torch.is_tensor(v)})


def measure(args, compute, inputs, batch, folder, guard, paced, reference):
    import torch
    c = config(); folder.mkdir()
    sampler = telemetry.TelemetrySampler(args.gpu_uuid)
    clocks = previous.ClockSampler(args.gpu_uuid)
    groups, updates, repeats = [], [], []
    sampler.start(); clocks.start()

    def wait(seconds):
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            guard.fast()
            time.sleep(min(.2, max(0, until - time.monotonic())))

    def phase(name, minimum, duty, adaptive=False):
        sampler.set_phase(name); start = time.monotonic(); last_update = start
        output = None
        while time.monotonic() - start < minimum:
            guard.fast(); tick = time.monotonic()
            for _ in range(8):
                output, raw = compute(inputs)
            torch.cuda.synchronize(); computed = time.monotonic()
            idle = imposed_wait(computed-tick, duty) if paced else 0.
            if idle: wait(idle)
            end = time.monotonic()
            groups.append(dict(phase=name, start_monotonic_s=tick, compute_end_monotonic_s=computed,
                end_monotonic_s=end, requested_idle_s=idle, duty=duty, forwards=8))
            if adaptive and end-last_update >= 2:
                recent = [s.utilization_percent for s in list(sampler.samples)
                          if max(start, end-2) <= s.host_monotonic_s < end]
                if len(recent) >= 10:
                    observed = sum(recent)/len(recent); new = update_duty(duty, observed)
                    updates.append(dict(at_monotonic_s=end, observed_percent=observed, old_duty=duty, new_duty=new))
                    duty = new
                last_update = end
        return start, time.monotonic(), output, duty

    try:
        duty = .5 if paced else 1.
        if paced:
            _, _, _, duty = phase("calibration", 30, duty, True)
        for repeat in range(1, 4):
            phase(f"warmup{repeat}", 5, duty)
            start, end, output, _ = phase(f"measured{repeat}", 30, duty)
            selected = [s for s in list(sampler.samples) if start <= s.host_monotonic_s < end]
            result = telemetry.telemetry_report(selected, prefix="")
            calls = sum(g["forwards"] for g in groups if g["phase"] == f"measured{repeat}")
            actual = output.detach().cpu()
            if not torch.isfinite(actual).all() or not torch.allclose(actual, reference, rtol=0, atol=.0002):
                raise RuntimeError("Frozen same-batch output changed during service")
            result.update(repeat=repeat, batch=batch, duration_s=end-start, start_monotonic_s=start,
                end_monotonic_s=end, forward_calls=calls, frozen_duty=duty, output=actual.tolist())
            if paced:
                result["service_cost"] = service_cost(end-start, calls, batch, result["mean_board_power_w"])
            old.write(folder/f"repeat{repeat:02d}.json", result); repeats.append(result)
        count = sum(r["samples"] for r in repeats)
        pooled = sum(r["samples"]*r["mean_gpu_utilization_percent"] for r in repeats)/count
        result = dict(status="complete", batch=batch, pid=os.getpid(), paced=paced, repeats=repeats,
            total_samples=count, pooled_mean_utilization_percent=pooled,
            target50_reached=abs(pooled-50) <= 5 if paced else None,
            frozen_duty=duty, no_window_or_repeat_selection=True,
            accuracy="No new full558 SRCC; repeated firstB output parity only")
        if paced:
            duration = sum(r["duration_s"] for r in repeats)
            board = sum(r["mean_board_power_w"]*r["duration_s"] for r in repeats)/duration
            result["pooled_service_cost"] = service_cost(duration, sum(r["forward_calls"] for r in repeats), batch, board)
        else:
            result["resident_power_not_dataset_once_cost"] = True
        old.write(folder/"report.json", result)
        return result
    finally:
        samples = sampler.stop()
        telemetry.save_samples(folder/"telemetry.csv", samples)
        clocks.stop(folder/"clock_telemetry.csv")
        old.write(folder/"forward_groups.json", groups)
        old.write(folder/"controller_updates.json", updates)


def worker(args):
    import torch
    c = config(); args.output = ROOT/args.model
    if args.output.exists() or not old.REPO.is_relative_to("/data"):
        raise RuntimeError("New canonical /data run required")
    args.output.mkdir(parents=True)
    identity = dict(sources=sources(), protocol=c, gpu_uuid=args.gpu_uuid, model=args.model,
        pid=os.getpid(), created=time.time(), command=sys.argv,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        same_model_process_all_batches=True, original_B1_B2_unchanged=True)
    guard = Guard(args, identity); reports = []; state = dict(status="loading", completed_batches=[])
    def save(): old.write(args.output/"status.json", dict(**state, time=time.time()))
    try:
        guard.check()
        raw = subprocess.check_output(["nvidia-smi", f"--id={args.gpu_uuid}",
            "--query-gpu=name,memory.free,power.limit,driver_version", "--format=csv,noheader,nounits"], text=True)
        name, free, watts, driver = next(csv.reader(raw.splitlines()))
        if "A100-SXM4-80GB" not in name or float(free) < 75000 or float(watts) != 400:
            raise RuntimeError("Exclusive free A10080 default400W required; hidden memory occupation is not safe")
        identity.update(gpu=raw.strip(), driver=driver.strip()); save()
        torch.set_num_threads(8)
        model, processor, prompt, first, compute = previous.runtime(args, identity, guard)
        if any(p.requires_grad for p in model.parameters()): raise RuntimeError("Frozen backbone required")
        old.write(args.output/"identity.json", identity); guard.start()
        data = old.rows(); test = [r for r in data if r["split"] == "test"]
        cpu, _ = previous.prepare_batch(args.model, [r for r in data if r["split"] == "train"][:1], processor, prompt)
        inputs = old.transfer(cpu); mos, tapped = compute(inputs)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            native = model(**inputs, use_cache=False, return_dict=True).logits[:, -1].float()
        if not torch.isfinite(native).all() or not torch.allclose(native, tapped, rtol=.01, atol=.125):
            raise RuntimeError("Native full vocabulary/feature tap parity failed")
        old.write(args.output/"native_parity.json", dict(train_videos=1, max_abs=float((native-tapped).abs().max()), rtol=.01, atol=.125))
        del cpu, inputs, mos, tapped, native
        # One audited TRAIN parity before the sequence; no cache clearing between batches.
        torch.cuda.empty_cache()
        for batch in range(1, c["safety_maximum_batch"]+1):
            if args.deadline-time.time() < c["reserve_after_ascending_seconds"]:
                state["ascending_stop"] = "finite_time_reserve_for_busy50"; break
            guard.fast(); state.update(status="ascending", batch=batch); save()
            cpu = inputs = reference = raw_logits = None
            try:
                cpu, positions = previous.prepare_batch(args.model, test[:batch], processor, prompt)
                folder = args.output/f"batch{batch:02d}"
                identity_input = input_identity(cpu, test[:batch], positions)
                old.write(args.output/f"batch{batch:02d}_input_identity.json", identity_input)
                inputs = old.transfer(cpu); reference, raw_logits = compute(inputs)
                reference = reference.detach().cpu(); del raw_logits
                torch.cuda.synchronize()
                report = measure(args, compute, inputs, batch, folder, guard, False, reference)
                old.write(folder/"input_identity.json", identity_input)
                reports.append(report); state["completed_batches"].append(batch); save()
                del cpu, inputs, reference
                print(json.dumps(dict(model=args.model, batch=batch, utilization=report["pooled_mean_utilization_percent"])), flush=True)
                if reached_target(report): state["ascending_stop"] = "pooled_mean_above98"; break
            except torch.cuda.OutOfMemoryError as error:
                old.write(args.output/f"batch{batch:02d}_capacity_failure.json", dict(error=str(error), no_retry=True, time=time.time()))
                cpu = inputs = reference = raw_logits = None
                error.__traceback__ = None
                import gc
                gc.collect()
                state["ascending_stop"] = "native_capacity_limit"; torch.cuda.empty_cache(); break
        if not reports: raise RuntimeError("No complete native batch measured")
        # Separate paced load only AFTER the uninterrupted ascending native sequence.
        for batch in sorted({1, reports[-1]["batch"]}):
            guard.fast(); state.update(status="busy50", batch=batch); save()
            cpu, positions = previous.prepare_batch(args.model, test[:batch], processor, prompt)
            inputs = old.transfer(cpu); reference, raw_logits = compute(inputs)
            reference = reference.detach().cpu(); del raw_logits
            folder = args.output/f"busy50_batch{batch:02d}"
            report = measure(args, compute, inputs, batch, folder, guard, True, reference)
            old.write(folder/"input_identity.json", input_identity(cpu, test[:batch], positions))
            del cpu, inputs, reference
            print(json.dumps(dict(model=args.model, paced_batch=batch, utilization=report["pooled_mean_utilization_percent"])), flush=True)
        if old.verify_assets(Path("/data/models")/args.model) != identity["assets"]:
            raise RuntimeError("Official model assets changed")
        guard.check(); state.update(status="complete")
        old.write(args.output/"report.json", dict(identity=identity, **state, native_batches=reports))
    except BaseException as error:
        state.update(status="failed", error=repr(error)); raise
    finally:
        guard.stop.set()
        if guard.thread: guard.thread.join(10)
        old.write(args.output/"process_audit.json", guard.audit); save()


def suite(args):
    if ROOT.exists(): raise RuntimeError("Do not overwrite existing run")
    ROOT.mkdir(parents=True)
    state = dict(status="running", completed=[], deadline=args.deadline, created=time.time(), sources=sources())
    try:
        # A blank container PID list does not imply free/exclusive device memory.
        until = min(args.deadline, time.time()+config()["maximum_initial_memory_wait_seconds"])
        stable = 0; memory_audit = []
        while stable < 3:
            if time.time() >= until:
                raise RuntimeError("Initial hidden-memory/foreign-process wait expired; no GPU work started")
            raw = subprocess.check_output(["nvidia-smi", f"--id={args.gpu_uuid}",
                "--query-gpu=memory.free,memory.used,power.limit", "--format=csv,noheader,nounits"], text=True)
            free, used, watts = map(float, next(csv.reader(raw.splitlines())))
            processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
                "--format=csv,noheader"], text=True)
            foreign = [r for r in csv.reader(processes.splitlines()) if len(r) == 2 and r[1].strip() == args.gpu_uuid]
            memory_audit.append(dict(time=time.time(), free_mib=free, used_mib=used, watts=watts, processes=foreign))
            old.write(ROOT/"initial_memory_audit.json", memory_audit)
            if watts != 400: raise RuntimeError("Do not change default power limit")
            stable = stable+1 if free >= 75000 and not foreign else 0
            state.update(status="waiting_for_free_memory_without_cuda", stable_checks=stable)
            old.write(ROOT/"suite_status.json", state)
            if stable < 3: time.sleep(10)
        state.update(status="running")
        for key in config()["models"]:
            state.update(phase=key); old.write(ROOT/"suite_status.json", state)
            # No CUDA context in the parent; each model gets one distinct process.
            command = [f"/data/venvs/{key}/bin/python", "-m", __package__+".rental_sequential_busy50",
                "--phase", "worker", "--model", key, "--gpu-uuid", args.gpu_uuid,
                "--deadline", str(min(args.deadline, time.time()+config()["maximum_model_seconds"]))]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu_uuid, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       TMPDIR="/data/tmp", HF_HOME="/data/cache")
            with (ROOT/(key+".log")).open("x") as log:
                child = subprocess.Popen(command, cwd=old.REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                state.update(child_pid=child.pid); old.write(ROOT/"suite_status.json", state)
                try: code = child.wait(timeout=max(1, min(args.deadline-time.time(), config()["maximum_model_seconds"])+5))
                except subprocess.TimeoutExpired:
                    import signal
                    os.killpg(child.pid, signal.SIGTERM)
                    try: child.wait(timeout=10)
                    except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait()
                    raise RuntimeError("Finite model deadline reached")
            if code: raise RuntimeError(f"{key} failed exit{code}; no automatic retry")
            state["completed"].append(key)
        state.update(status="complete", finished=time.time())
    except BaseException as error:
        state.update(status="failed", error=repr(error)); raise
    finally: old.write(ROOT/"suite_status.json", state)


def capacity_check(args):
    """One Kimi TRAIN B1 feasibility probe, never formal cost or utilization."""
    if args.model != "kimi":
        raise ValueError("Only one Kimi B1 diagnostic; DS resident weights already exceed free memory")
    args.output = old.RUN/"capacity_check_20261018"/"kimi"
    if args.output.exists(): raise RuntimeError("Single diagnostic only; preserve prior evidence")
    args.output.mkdir(parents=True)
    identity = dict(sources=sources(), model=args.model, gpu_uuid=args.gpu_uuid,
        pid=os.getpid(), created=time.time(), command=sys.argv,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        purpose="capacity_only_not_performance_measurement", no_cost_or_utilization_claim=True)
    guard = Guard(args, identity)
    state = dict(status="loading", no_formal_measurement=True)
    try:
        guard.check()
        import torch
        torch.set_num_threads(8)
        free, total = torch.cuda.mem_get_info()
        # Separate diagnostic threshold, never replaces the formal 75000MiB gate.
        required_weight_bytes = 32815315552
        if free < required_weight_bytes + 8*1024**3:
            raise RuntimeError("Insufficient free memory even for bounded Kimi B1 capacity probe")
        identity.update(free_before_bytes=free, total_bytes=total,
                        required_weight_bytes=required_weight_bytes, diagnostic_margin_bytes=8*1024**3)
        old.write(args.output/"status.json", state)
        model, processor, prompt, first, compute = previous.runtime(args, identity, guard)
        old.write(args.output/"identity.json", identity); guard.start()
        train = [r for r in old.rows() if r["split"] == "train"][:1]
        cpu, positions = previous.prepare_batch("kimi", train, processor, prompt)
        inputs = old.transfer(cpu)
        mos, tapped = compute(inputs)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            native = model(**inputs, use_cache=False, return_dict=True).logits[:, -1].float()
        torch.cuda.synchronize()
        if not torch.isfinite(native).all() or not torch.isfinite(mos).all():
            raise RuntimeError("Nonfinite native capacity probe")
        if not torch.allclose(native, tapped, rtol=.01, atol=.125):
            raise RuntimeError("Native full-vocabulary/tap parity failed")
        if any(p.requires_grad for p in model.parameters()):
            raise RuntimeError("Frozen original model required")
        old.write(args.output/"input_identity.json", input_identity(cpu, train, positions))
        result = dict(status="passed", identity=identity, batch=1, train_videos=1,
            native_logit_max_abs=float((native-tapped).abs().max()), rtol=.01, atol=.125,
            output_mos=mos.detach().cpu().tolist(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            cuda_free_after_forward_bytes=torch.cuda.mem_get_info()[0],
            formal_utilization_or_cost=False, full558_accuracy=False,
            limitation="Unattributed baseline VRAM remains; this only proves this one input executed")
        if old.verify_assets(Path("/data/models/kimi")) != identity["assets"]:
            raise RuntimeError("Official frozen assets changed")
        guard.check(); old.write(args.output/"report.json", result)
        state.update(status="complete")
    except BaseException as error:
        state.update(status="failed", error=repr(error)); raise
    finally:
        guard.stop.set()
        if guard.thread: guard.thread.join(10)
        old.write(args.output/"process_audit.json", guard.audit)
        old.write(args.output/"status.json", dict(**state, time=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("suite", "worker", "capacity-check"), required=True)
    parser.add_argument("--model", choices=("deepseek", "kimi"))
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    if not old.REPO.is_relative_to("/data") or not 0 < args.deadline-time.time() <= config()["maximum_suite_seconds"]:
        raise RuntimeError("New /data run within this one-hour rental, backup reserve required")
    if args.phase != "suite" and os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid:
        raise RuntimeError("Exact authorized GPU UUID required")
    {"suite": suite, "worker": worker, "capacity-check": capacity_check}[args.phase](args)


if __name__ == "__main__": main()
