"""Bounded CLIP saturation extension and native Qwen execution diagnosis.

Reuse the historical measurement workers without changing their loops, weights,
timing boundaries, precision or power. Profiler traces are diagnostic only.
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from . import measure_frozen_batch_costs as costs
from . import unified_resident_utilization as resident
from . import evaluate_temporal_baseline_batches as frozen
from . import run_temporal_telemetry_suite as prior
from . import reproduce_temporal_baseline as reproduction

TASK, REPO = prior.TASK, prior.REPO
CONFIG = TASK / "configs/baselines/clip_saturation_qwen_plateau_20261016.json"


def config():
    c = json.loads(CONFIG.read_text(encoding="utf-8"))
    if c["clip_candidates"] != list(range(11, 17)) or c["qwen_diagnostic_batches"] != [2, 4, 8, 10]:
        raise ValueError("Only the explicitly bounded extension is authorized")
    if c["target_mean_percent"] != 98 or c["maximum_suite_s"] != 3600 or c["maximum_initial_idle_wait_s"] != 1800:
        raise ValueError("Finite budget and target must not change")
    if c["gpu_uuid"] != frozen.A100_UUID or c["power_limit_w"] != 250 or not c["no_training_or_graph_or_power_changes"]:
        raise ValueError("Original hardware/model contract required")
    return c


def sources():
    return {**costs.sources(), str(Path(__file__).relative_to(REPO)): frozen.sha256(Path(__file__)),
            str(CONFIG.relative_to(REPO)): frozen.sha256(CONFIG)}


def extension_protocol(kind, batch=None):
    c = copy.deepcopy(resident.config() if kind == "util" else costs.config())
    c["batches"] = {"clip": config()["clip_candidates"] if kind == "util" else [batch]}
    c["extension_authorization"] = config()["authorization"]
    c["extension_source_sha256"] = sources()
    return c


def selected_cost_batches(points):
    """Measure the first threshold crossing and preceding NEW tested batch."""
    if not points:
        raise ValueError("No complete CLIP point")
    batches = [p["batch"] for p in points]
    if batches != config()["clip_candidates"][:len(batches)]:
        raise ValueError("Ascending authorized prefix required")
    return batches[-2:]


def union_duration(intervals):
    merged, total = None, 0.0
    for a, b in sorted(intervals):
        if b < a:
            raise ValueError("Negative interval")
        if merged is None:
            merged = [a, b]
        elif a <= merged[1]:
            merged[1] = max(b, merged[1])
        else:
            total += merged[1]-merged[0]
            merged = [a, b]
    return total + (merged[1]-merged[0] if merged else 0)


def trace_summary(trace):
    events = trace["traceEvents"]
    kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") == "kernel" and e.get("dur", 0) > 0]
    if not kernels:
        raise RuntimeError("No CUDA kernel trace; do not infer a bottleneck")
    a, b = min(e["ts"] for e in kernels), max(e["ts"]+e["dur"] for e in kernels)
    busy = union_duration((e["ts"], e["ts"]+e["dur"]) for e in kernels)
    api = collections.defaultdict(lambda: [0, 0.0])
    for e in events:
        if e.get("ph") == "X" and e.get("cat") in ("cuda_runtime", "cuda_driver"):
            api[e["name"]][0] += 1
            api[e["name"]][1] += e.get("dur", 0)
    copies = [e for e in events if e.get("ph") == "X" and "gpu_memcpy" in e.get("cat", "")]
    return dict(kernel_count=len(kernels), first_to_last_kernel_us=b-a, kernel_union_us=busy,
                kernel_gap_us=b-a-busy, kernel_union_fraction=busy/(b-a),
                cuda_api=[dict(name=k, calls=v[0], total_us=v[1]) for k, v in sorted(api.items(), key=lambda kv: -kv[1][1])],
                device_copy_events=[dict(name=e["name"], duration_us=e.get("dur", 0)) for e in copies],
                interpretation="Profiler-instrumented union of kernel intervals, not NVML utilization or SM occupancy; CPU/API inclusive times are not additive")


def qwen_profile(args):
    frozen.preflight_files(args)
    reproduction.cuda_guard()
    import torch
    from . import quality_token_common as core
    from . import audit_utilization_execution as execution
    from . import measure_resident_utilization as old_resident
    from .diagnose_batch_numerics import input_records
    from LightGenV2.common.baseline_measurement import gpu_power_limit_w

    if abs(gpu_power_limit_w()-250) > .1:
        raise RuntimeError("Unchanged250W required")
    args.output.mkdir()
    bound = sources()
    assets = costs.model_hashes(args.model)
    frozen.write_json(args.output / "identity.json", dict(source_sha256=bound, model_sha256=assets,
        checkpoint_sha256=frozen.sha256(args.checkpoint), manifest_sha256=frozen.sha256(args.manifest),
        gpu_uuid=frozen.A100_UUID, protocol=config(), command=sys.argv, pid=os.getpid(),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        profiler_results_are_not_formal_costs=True))
    timer, handles = None, []
    try:
        with torch.inference_mode():
            model, head, _, prepare, transfer, native, timer, handles = frozen.load_runtime(args)
            compute = execution.make_gpu_forward(args, model, head, native)
            if any(p.requires_grad for m in (model, head) for p in m.parameters()):
                raise RuntimeError("Frozen weights required")
            rows = [r for r in core.read_manifest(args.manifest) if r["split"] == "test"]
            result = dict(status="running", points=[])
            for batch in config()["qwen_diagnostic_batches"]:
                if sources() != bound or time.time() >= args.deadline:
                    raise RuntimeError("Source/deadline guard")
                frozen.process_audit()
                folder = args.output / f"batch{batch:02d}"
                folder.mkdir()
                cpu, positions = prepare(rows[:batch])
                records = input_records(cpu, batch)
                inputs = transfer(cpu)
                torch.cuda.synchronize()
                if batch == config()["qwen_diagnostic_batches"][0]:
                    ref = torch.tensor(native(inputs)[0])
                    timer.close()
                else:
                    ref = compute(inputs).detach().cpu()
                first = compute(inputs).detach().cpu()
                if not torch.isfinite(first).all() or not torch.allclose(first, ref, atol=.0002, rtol=0):
                    raise RuntimeError("Native output parity failed")
                baseline = resident.measure_repeat(args, compute, inputs, folder / "unprofiled_repeat01", resident.config())
                guard = old_resident.ForeignGuard()
                guard.start()
                try:
                    guard.check()
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
                        for _ in range(config()["profile_forward_calls"]):
                            with torch.profiler.record_function("native_resident_forward"):
                                output = compute(inputs)
                        torch.cuda.synchronize()
                    guard.check()
                finally:
                    guard.stop()
                    frozen.write_json(folder / "profile_process_audit.json", guard.records)
                if not torch.allclose(output.detach().cpu(), ref, atol=.0002, rtol=0):
                    raise RuntimeError("Profiler changed native MOS")
                target = folder / "native_trace.json"
                prof.export_chrome_trace(str(target))
                if target.stat().st_size > config()["profile_maximum_bytes"]:
                    raise RuntimeError("Trace exceeded bounded storage; preserve evidence")
                measured = trace_summary(json.loads(target.read_text()))
                stats = [dict(name=e.key, calls=e.count, self_cpu_time_us=e.self_cpu_time_total,
                              inclusive_cpu_time_us=e.cpu_time_total) for e in prof.key_averages()]
                entry = dict(batch=batch, unprofiled_measurement=baseline["measurement"],
                    profiled=measured, top_self_cpu_operators=sorted(stats,key=lambda x:-x["self_cpu_time_us"])[:40],
                    input_records=records, sample_ids=[r["sample_id"] for r in rows[:batch]],
                    frame_positions=positions, native_mos=ref.tolist(), output_parity_max_abs=float((output.detach().cpu()-ref).abs().max()))
                frozen.write_json(folder / "report.json", entry)
                result["points"].append(entry)
                frozen.write_json(args.output / "report.json", result)
                print(json.dumps(dict(batch=batch, native_util=baseline["measurement"]["mean_gpu_utilization_percent"],
                                      profile_kernel_union_fraction=measured["kernel_union_fraction"])), flush=True)
                del inputs, cpu, prof
            if sources() != bound or costs.model_hashes(args.model) != assets:
                raise RuntimeError("Bound source/assets changed")
            result.update(status="complete", completed_unix=time.time())
            frozen.write_json(args.output / "report.json", result)
    finally:
        if timer is not None:timer.close()
        for h in handles:h.remove()


def child_command(kind, args, output, batch=1):
    baseline = "qwen" if kind == "qwen-profile" else "clip"
    command = prior.child_command(baseline, batch, args.manifest, output)
    command[command.index(__package__+".evaluate_temporal_baseline_batches")] = __package__+".extend_clip_diagnose_qwen"
    command.remove("--telemetry")
    return command+["--phase", kind, "--deadline", str(args.deadline)]


def suite(args):
    import fcntl
    reproduction.require_new_output(args.output)
    args.output.mkdir()
    state = dict(status="waiting_for_idle", deadline=args.deadline, pid=os.getpid(), completed=[],
                 protocol=config(), source_sha256=sources(), command=sys.argv)
    def save():frozen.write_json(args.output / "suite_status.json", state)
    save()
    with (TASK / "runs/simulation/.temporal_telemetry_a100.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            costs.wait_for_idle(args, {"maximum_initial_idle_wait_seconds":config()["maximum_initial_idle_wait_s"]})
            bound = sources()
            def dispatch(kind, name, batch=1):
                reproduction.cuda_guard()
                if sources() != bound:raise RuntimeError("Bound suite source changed")
                state.update(status="running", phase=name, updated_unix=time.time());save()
                command=child_command(kind,args,args.output/name,batch)
                frozen.write_json(args.output/(name+".command.json"),command)
                env=dict(os.environ,HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
                limit=config()["maximum_qwen_diagnostic_s"] if kind=="qwen-profile" else config()["maximum_worker_s"]
                with (args.output/(name+".log")).open("x") as stream:
                    subprocess.run(command,cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True,
                                   timeout=min(limit,args.deadline-time.time()))
                r=json.loads((args.output/name/"report.json").read_text())
                if r["status"]!="complete":raise RuntimeError("Incomplete phase")
                state["completed"].append(name);save()
            dispatch("clip-util","clip_utilization")
            points=json.loads((args.output/"clip_utilization/report.json").read_text())["points"]
            state["selected_cost_batches"]=selected_cost_batches(points);save()
            for batch in state["selected_cost_batches"]:
                dispatch("clip-cost",f"clip_cost_batch{batch:02d}",batch)
            dispatch("qwen-profile","qwen_execution_profile")
            reproduction.cuda_guard()
            state.update(status="complete",completed_unix=time.time())
        except BaseException as error:
            state.update(status="failed",error=repr(error),observed_unix=time.time(),no_automatic_retry=True)
            raise
        finally:save()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase",choices=("suite","clip-util","clip-cost","qwen-profile"),required=True)
    p.add_argument("--baseline",choices=("qwen","clip"))
    p.add_argument("--batch",type=int,default=1)
    for name in ("manifest","output","model","checkpoint","reproduction-receipt"):
        p.add_argument("--"+name,type=Path,required=name in ("manifest","output"))
    p.add_argument("--deadline",type=float,required=True)
    args=p.parse_args();args.output=args.output.resolve()
    if os.environ.get("CUDA_VISIBLE_DEVICES")!=frozen.A100_UUID or frozen.sha256(args.manifest)!=frozen.MANIFEST_SHA:
        raise RuntimeError("Original A100/manifest required")
    if not 0<args.deadline-time.time()<=config()["maximum_suite_s"]:
        raise ValueError("Bounded one-hour absolute deadline required")
    if args.phase=="suite":suite(args)
    else:
        expected="qwen" if args.phase=="qwen-profile" else "clip"
        if args.baseline!=expected:raise ValueError("Wrong model for phase")
        if frozen.sha256(args.checkpoint)!=costs.CHECKPOINTS[expected]:raise RuntimeError("Original frozen PT required")
        frozen.write_json(args.output.with_suffix(".extension_identity.json"),dict(command=sys.argv,source_sha256=sources(),protocol=config()))
        if args.phase=="clip-util":resident.worker(args,extension_protocol("util"))
        elif args.phase=="clip-cost":
            if args.batch not in config()["clip_candidates"]:raise ValueError("Only bounded new CLIP batches")
            costs.worker(args,extension_protocol("cost",args.batch))
        else:qwen_profile(args)


if __name__=="__main__":main()
