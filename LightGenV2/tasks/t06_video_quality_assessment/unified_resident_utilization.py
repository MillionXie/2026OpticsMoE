"""One resident native inference method, three identical repeats, six frozen models.

This is utilization-only. Historical costs/accuracy and the finalized Qwen B1
reference are never overwritten. No Graph, model patches, training or offload.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import threading
import time

from . import evaluate_temporal_baseline_batches as frozen
from . import reproduce_temporal_baseline as reproduction
from . import run_temporal_telemetry_suite as prior
from . import measure_resident_utilization as resident
from . import audit_utilization_execution as execution
from . import batch_telemetry as telemetry

TASK = Path(__file__).resolve().parent
REPO = TASK.parents[2]
CONFIG = TASK / "configs/baselines/unified_resident_utilization_20261015.json"
SPECIAL = {
    "glm": ("lgvq_glm46v_flash_temporal_20261015/head_width_recovery01", "glm46v_flash_temporal_baseline",
            "/DATA/DATA1/guest3/.venvs/lgvq-glm46v/bin/python"),
    "small": ("lgvq_deepseek_vl2_small_temporal_20261015/assetio_recovery01", "deepseek_vl2_small_temporal_baseline",
              "/DATA/DATA1/guest3/.venvs/deepseek-vl2/bin/python"),
}


def config():
    c = json.loads(CONFIG.read_text())
    if (c["warmup_s"], c["measurement_s"], c["repeats"], c["queue_depth"]) != (5, 30, 3, 8):
        raise ValueError("Fixed measurement method changed")
    if c["cuda_graph"] or not c["historical_costs_and_metrics_unchanged"]:
        raise ValueError("Graph/cost substitution prohibited")
    for plan in c["batches"].values():
        if plan[:3] != [1, 2, 3] or sorted(set(plan)) != plan or max(plan) > 10:
            raise ValueError("Bounded small ascending batches required")
    return c


def sources():
    files = [Path(__file__), CONFIG, Path(frozen.__file__), Path(resident.__file__),
             Path(execution.__file__), Path(telemetry.__file__), Path(prior.__file__),
             TASK / "glm46v_flash_temporal_baseline.py", TASK / "deepseek_vl2_small_temporal_baseline.py",
             TASK / "quality_token_common.py", TASK / "quality_token_batch_benchmark_200.py",
             TASK / "deepseek_vl2_quality.py", TASK / "configs/baselines/glm46v_flash_temporal_4f.json",
             TASK / "configs/baselines/deepseek_vl2_small_temporal_4f.json"]
    return {str(p.relative_to(REPO)): frozen.sha256(p) for p in files}


def cpu_snapshot():
    import resource
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result = dict(monotonic_s=time.monotonic(), user_s=usage.ru_utime, system_s=usage.ru_stime,
                  voluntary_switches=usage.ru_nvcsw, involuntary_switches=usage.ru_nivcsw,
                  affinity=sorted(os.sched_getaffinity(0)))
    p = Path("/proc/self/schedstat")
    if p.exists():
        result["main_thread_schedstat"] = list(map(int, p.read_text().split()))
    return result


class ClockSampler:
    """Separate raw clock/temperature observations; never alter clock settings."""
    def start(self):
        self.rows = []
        self.process = subprocess.Popen(["nvidia-smi", f"--id={frozen.A100_UUID}",
            "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,pstate,power.limit",
            "--format=csv,noheader,nounits", "--loop-ms=200"], stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True)
        def read():
            for line in self.process.stdout:
                values = [x.strip() for x in line.split(",")]
                if len(values) == 5:
                    self.rows.append([time.time(), time.monotonic(), *values])
        self.thread = threading.Thread(target=read, daemon=True)
        self.thread.start()

    def stop(self, path):
        self.process.terminate()
        self.process.wait(timeout=5)
        self.thread.join(timeout=5)
        if not self.rows:
            raise RuntimeError("Clock telemetry missing")
        with path.open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["host_time_unix", "host_monotonic_s", "sm_clock_mhz", "memory_clock_mhz",
                             "temperature_c", "pstate", "power_limit_w"])
            writer.writerows(self.rows)


def pooled_mean(repeats):
    return sum(x["measurement"]["mean_gpu_utilization_percent"] * x["measurement"]["samples"]
               for x in repeats) / sum(x["measurement"]["samples"] for x in repeats)


def stop_at_target(batch, value, c):
    # Explicit reproduction profiles must reach the user-requested terminal B.
    if batch < c.get("required_terminal_batch", 0):
        return False
    return batch >= 2 and value >= c["target_mean_percent"]


def measure_repeat(args, compute, inputs, folder, c):
    import torch
    folder.mkdir()
    sampler, clocks, guard = telemetry.TelemetrySampler(frozen.A100_UUID), ClockSampler(), resident.ForeignGuard()
    groups, samples = [], []
    sampler.start(); clocks.start(); guard.start()
    def phase(name, minimum):
        sampler.set_phase(name)
        started = time.monotonic()
        while time.monotonic() - started < minimum:
            if time.time() >= args.deadline:
                raise RuntimeError("Original suite deadline reached")
            guard.check()
            tick = time.monotonic()
            for _ in range(c["queue_depth"]):
                output = compute(inputs)
            torch.cuda.synchronize()
            groups.append(dict(phase=name, start_monotonic_s=tick,
                               end_monotonic_s=time.monotonic(), forwards=c["queue_depth"]))
        return started, time.monotonic(), output
    try:
        before = cpu_snapshot()
        a, b, _ = phase("warmup", c["warmup_s"])
        cpu_start = cpu_snapshot()
        start, end, output = phase("measured", c["measurement_s"])
        cpu_end = cpu_snapshot()
        sampler.set_phase("after")
        samples = sampler.stop()
        guard.stop(); guard.check()
        result = dict(warmup=resident.summarize(samples, a, b),
                      measurement=resident.summarize(samples, start, end),
                      cpu_before_warmup=before, cpu_measurement_start=cpu_start, cpu_measurement_end=cpu_end,
                      measured_forward_invocations=sum(g["forwards"] for g in groups if g["phase"] == "measured"),
                      output=output.detach().cpu().tolist())
        frozen.write_json(folder / "report.json", result)
        return result
    finally:
        if sampler.process is not None:
            samples = sampler.stop()
        guard.stop()
        clocks.stop(folder / "clock_telemetry.csv")
        telemetry.save_samples(folder / "telemetry.csv", samples)
        frozen.write_json(folder / "forward_groups.json", groups)
        frozen.write_json(folder / "process_audit.json", guard.records)


def special_runtime(args):
    import importlib
    import torch
    module = importlib.import_module(__package__ + "." + SPECIAL[args.baseline][1])
    original = TASK / "runs/simulation" / SPECIAL[args.baseline][0]
    report = json.loads((original / "a100_batch01/report.json").read_text())
    receipt = json.loads((original / "train/training_receipt.json").read_text())
    args.checkpoint = original / "train/best_checkpoint.pt"
    command = report["identity"]["command"]
    args.model = Path(command[command.index("--model")+1])
    if frozen.sha256(args.checkpoint) != report["checkpoint_sha256"] or receipt["artifacts"]["best_checkpoint.pt"] != report["checkpoint_sha256"]:
        raise RuntimeError("Fixed trained checkpoint identity changed")
    # Full official hashes, including weights, before and after the entire model series.
    assets = (module.verify_model_assets(args, module.config(), "before_GPU") if args.baseline == "small"
              else module.shared.model_files(args.model, module.config()))
    if assets != receipt["identity"]["model_sha256"] or module.runtime_sources() != receipt["identity"]["runtime_sha256"]:
        raise RuntimeError("Official model/runtime differs from training receipt")
    model, ids = module.native_model(args.model)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if ids != payload["quality_token_ids"]:
        raise ValueError("Quality tokens changed")
    head = module.five_native_rows(payload["state_dict"]["weight"]).cuda().eval().requires_grad_(False)
    scores = payload["level_scores"].cuda()
    def prepare(rows):
        if args.baseline == "glm" or len(rows) == 1:
            return module.prepare_batch(rows, model)
        from . import deepseek_vl2_quality as tiny
        from . import quality_token_common as core
        prepared, positions = [], []
        for row in rows:
            frames, where = tiny.decode_ordered_frames(Path(row["video_path"]), core.FRAME_FRACTIONS[4], 384)
            prepared.append(model._quality_processor.process_one(conversations=tiny.build_conversation("temporal", 4),
                            images=frames, system_prompt=""))
            positions.append(where)
        packed = model._quality_processor.batchify(prepared)
        return {k: packed[k] for k in ("input_ids", "attention_mask", "images", "images_seq_mask", "images_spatial_crop")}, positions
    def transfer(inputs):
        return module.transfer(inputs) if args.baseline == "small" else module.shared.transfer(inputs)
    def compute(inputs):
        # Same native feature arithmetic and FP32 readout. Finite-value audits
        # run outside the timed loop instead of adding an audit-only CPU sync.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if args.baseline == "glm":
                hidden = model.model(**inputs, use_cache=False, return_dict=True).last_hidden_state
            else:
                embeds = model.prepare_inputs_embeds(**inputs)
                hidden = model.language.model(inputs_embeds=embeds, attention_mask=inputs["attention_mask"],
                                              use_cache=False, return_dict=True).last_hidden_state
            mask = inputs["attention_mask"].bool()
            last = torch.arange(mask.shape[1], device=hidden.device).expand_as(mask).masked_fill(~mask, -1).amax(1)
            features = hidden[torch.arange(len(hidden), device=hidden.device), last].float()
        return head(features).softmax(-1) @ scores
    def reference(inputs):
        if args.baseline == "small" and inputs["input_ids"].shape[0] != 1:
            # The original accuracy entry deliberately admitted only B1. The
            # unmodified native batched graph is used here; do not call its
            # B1-only shape assertion or claim a separate B>1 accuracy audit.
            return compute(inputs)
        return head(module.features(model, inputs)).softmax(-1) @ scores
    def finish():
        after = (module.verify_model_assets(args, module.config(), "after_GPU") if args.baseline == "small"
                 else module.shared.model_files(args.model, module.config()))
        if after != assets:
            raise RuntimeError("Official model assets changed")
    return model, head, prepare, transfer, compute, reference, finish


def worker(args, c):
    import torch
    from . import quality_token_common as core
    from .diagnose_batch_numerics import input_records
    from LightGenV2.common.baseline_measurement import gpu_power_limit_w
    if abs(gpu_power_limit_w()-250) > .1:
        raise RuntimeError("Unchanged250W power limit required")
    frozen.process_audit()
    # Existing-output protection must run before creating the worker directory.
    if args.baseline not in SPECIAL:
        frozen.preflight_files(args)
    args.output.mkdir()
    identity = dict(protocol=c, baseline=args.baseline, pid=os.getpid(), gpu_uuid=frozen.A100_UUID,
                    source_sha256=sources(), source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                    manifest_sha256=frozen.sha256(args.manifest), torch=torch.__version__, python=platform.python_version(),
                    cpu_initial=cpu_snapshot(), allocator=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
    frozen.write_json(args.output / "identity.json", identity)
    with torch.inference_mode():
        if args.baseline in SPECIAL:
            model, head, prepare, transfer, compute, reference, finish = special_runtime(args)
        else:
            model, head, _, prepare, transfer, native, timer, handles = frozen.load_runtime(args)
            compute = execution.make_gpu_forward(args, model, head, native)
            native_checked = False
            def reference(inputs):
                nonlocal native_checked
                if not native_checked:
                    values = torch.tensor(native(inputs)[0], device="cuda")
                    timer.close()
                    native_checked = True
                    return values
                return compute(inputs)
            finish = lambda: None
        if any(p.requires_grad for m in (model, head) for p in m.parameters()):
            raise RuntimeError("All weights must be frozen")
        identity["checkpoint_sha256"] = frozen.sha256(args.checkpoint)
        frozen.write_json(args.output / "identity.json", identity)
        rows = [r for r in core.read_manifest(args.manifest) if r["split"] == "test"]
        state = dict(status="running", identity=identity, points=[])
        frozen.write_json(args.output / "report.json", state)
        for batch in c["batches"][args.baseline]:
            frozen.process_audit()
            if identity["source_sha256"] != sources():
                raise RuntimeError("Bound measurement source changed")
            point = args.output / f"batch{batch:02d}"
            point.mkdir()
            args.batch = batch
            try:
                cpu, positions = prepare(rows[:batch])
                fingerprints = input_records(cpu, batch) if args.baseline not in SPECIAL else None
                fields = (cpu if isinstance(cpu, dict) else ({"frames":cpu} if torch.is_tensor(cpu)
                          else {k:v for k,v in vars(cpu).items() if torch.is_tensor(v)}))
                tensors = {k:dict(shape=list(v.shape), dtype=str(v.dtype), sha256=hashlib.sha256(
                    v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
                    for k,v in fields.items()}
                if not tensors:
                    raise RuntimeError("Input tensor identity missing")
                inputs = transfer(cpu)
                del cpu
                torch.cuda.synchronize()
                ref = reference(inputs).detach().cpu()
                first = compute(inputs).detach().cpu()
                if not torch.isfinite(first).all() or not torch.allclose(first, ref, rtol=0, atol=.0002):
                    raise RuntimeError("Native MOS parity failed")
                original_b1 = None
                if batch == 1 and args.baseline in SPECIAL:
                    original_csv = TASK / "runs/simulation" / SPECIAL[args.baseline][0] / "a100_batch01/predictions.csv"
                    with original_csv.open() as stream:
                        original_b1 = next(csv.DictReader(stream))
                    if (original_b1["sample_id"] != rows[0]["sample_id"]
                            or abs(float(original_b1["prediction"])-float(first[0])) > .0002):
                        raise RuntimeError("B1 output differs from finalized native accuracy record")
                # A single admission probe above is not part of warmup or measurement.
                repeats = []
                for index in range(c["repeats"]):
                    frozen.write_json(args.output / "status.json", dict(status="measuring", batch=batch,
                        repeat=index+1, completed_points=len(state["points"]), updated_unix=time.time()))
                    repeat = measure_repeat(args, compute, inputs, point / f"repeat{index+1:02d}", c)
                    if max(abs(a-b) for a,b in zip(repeat["output"], ref.tolist())) > .0002:
                        raise RuntimeError("Resident MOS changed")
                    repeats.append(repeat)
                means = [r["measurement"]["mean_gpu_utilization_percent"] for r in repeats]
                result = dict(status="complete", batch=batch, sample_ids=[r["sample_id"] for r in rows[:batch]],
                    frame_positions=positions, input_fingerprints=fingerprints,
                    input_tensor_identity=tensors, reference_mos=ref.tolist(), repeats=repeats,
                    original_b1_reference=original_b1,
                    mean_utilization_percent=pooled_mean(repeats), repeat_means_percent=means,
                    repeat_std_pp=statistics.stdev(means), repeat_spread_pp=max(means)-min(means),
                    unstable_repeat_warning=max(means)-min(means)>c["repeat_spread_warning_pp"],
                    peak_memory_allocated_bytes=torch.cuda.max_memory_allocated())
                frozen.write_json(point / "report.json", result)
                state["points"].append(dict(batch=batch, mean=result["mean_utilization_percent"], repeat_means=means,
                                            unstable_repeat_warning=result["unstable_repeat_warning"]))
                frozen.write_json(args.output / "report.json", state)
                print(json.dumps(state["points"][-1]), flush=True)
                del inputs, ref, first
                if stop_at_target(batch, result["mean_utilization_percent"], c):
                    state["stop_reason"] = "first ascending batch >=98%; preceding tested batch retained"
                    break
            except torch.cuda.OutOfMemoryError as error:
                frozen.write_json(point / "capacity_limit.json", dict(batch=batch, error=str(error),
                    no_retry=True, no_quantization_or_offload=True))
                state["stop_reason"] = f"native capacity limit at batch{batch}; no retry"
                state["capacity_limited"] = True
                break
        finish()
        if abs(gpu_power_limit_w()-250) > .1:
            raise RuntimeError("Power limit changed")
        if identity["checkpoint_sha256"] != frozen.sha256(args.checkpoint):
            raise RuntimeError("Trained checkpoint changed")
        state.update(status="complete", completed_unix=time.time())
        frozen.write_json(args.output / "report.json", state)
        frozen.write_json(args.output / "status.json", dict(status="complete", updated_unix=time.time()))


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
        hashes = sources()
        state = dict(status="running", pid=os.getpid(), started_unix=time.time(), deadline=args.deadline,
                     protocol=c, source_sha256=hashes, completed=[], failed=[])
        for baseline in (args.models or c["batches"]):
            if time.time() >= args.deadline or hashes != sources():
                raise RuntimeError("Suite deadline/source guard")
            reproduction.cuda_guard()
            state["phase"] = baseline
            frozen.write_json(args.output / "suite_status.json", state)
            output = args.output / baseline
            if baseline in SPECIAL:
                command = [SPECIAL[baseline][2], "-u", "-m", __package__ + ".unified_resident_utilization",
                           "--phase", "worker", "--baseline", baseline, "--manifest", str(args.manifest), "--output", str(output)]
            else:
                command = prior.child_command(baseline, 1, args.manifest, output)
                command[command.index(__package__ + ".evaluate_temporal_baseline_batches")] = __package__ + ".unified_resident_utilization"
                command.remove("--telemetry"); command.extend(["--phase", "worker"])
            command.extend(["--deadline", str(args.deadline)])
            frozen.write_json(args.output / (baseline + ".command.json"), command)
            environment = dict(os.environ)
            if baseline == "small":
                environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            with (args.output / (baseline + ".log")).open("x") as stream:
                process = subprocess.Popen(command, cwd=REPO, env=environment, stdout=stream,
                    stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    returncode = process.wait(timeout=max(1, args.deadline-time.time()))
                except subprocess.TimeoutExpired:
                    import signal
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=10)
                    returncode = 124
            if returncode:
                state["failed"].append(dict(baseline=baseline, returncode=returncode))
            else:
                state["completed"].append(baseline)
            frozen.write_json(args.output / "suite_status.json", state)
            reproduction.cuda_guard()
        state.update(status="complete" if not state["failed"] else "partial_failure", completed_unix=time.time())
        frozen.write_json(args.output / "suite_status.json", state)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("suite", "worker"), required=True)
    p.add_argument("--baseline", choices=tuple(config()["batches"]))
    p.add_argument("--models", choices=tuple(config()["batches"]), nargs="+",
                   help="Continue only explicitly selected unmeasured models, keeping the original absolute deadline")
    for name in ("manifest", "output", "model", "checkpoint", "reproduction-receipt"):
        p.add_argument("--" + name, type=Path, required=name in ("manifest", "output"))
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--deadline", type=float, required=True)
    args = p.parse_args()
    if args.models and len(set(args.models)) != len(args.models):
        p.error("Duplicate model selection")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != frozen.A100_UUID or frozen.sha256(args.manifest) != frozen.MANIFEST_SHA:
        raise RuntimeError("Only approved A100 and original split")
    if not 0 < args.deadline-time.time() <= config()["maximum_suite_s"]:
        raise ValueError("Explicit bounded two-hour deadline required")
    args.output = args.output.resolve()
    try:
        if args.phase == "suite":
            suite(args, config())
        else:
            if not args.baseline:
                p.error("Worker baseline required")
            worker(args, config())
    except BaseException as error:
        if args.output.is_dir():
            frozen.write_json(args.output / "failure.json", dict(status="failed", error=repr(error),
                pid=os.getpid(), observed_unix=time.time(), no_automatic_retry=True))
        raise


if __name__ == "__main__":
    main()
