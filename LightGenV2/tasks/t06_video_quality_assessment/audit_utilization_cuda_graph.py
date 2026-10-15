"""One bounded native CUDA Graph attempt per fixed case; never paper data.

Fixed resident TEST inputs, three side-stream warmup calls, one capture,
30-second graph replay, fixed full and 5-30s telemetry. No capture retries,
model patches, fake kernels, changed precision, or historical cost replacement.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from . import audit_utilization_execution as controls
from . import evaluate_temporal_baseline_batches as frozen
from . import batch_telemetry as telemetry
from . import reproduce_temporal_baseline as reproduction

CASES = (("qwen", 8), ("clip", 10), ("yolo", 3))
REPLAY_GROUP = 16
WARMUP_CALLS = 3


def run(args):
    import torch
    if os.environ.get("CUDA_VISIBLE_DEVICES") != frozen.A100_UUID:
        raise RuntimeError("Wrong physical GPU")
    reproduction.cuda_guard()
    frozen.preflight_files(args)
    args.output.mkdir(parents=True, exist_ok=False)
    identity = dict(status="loading", baseline=args.baseline, batch=args.batch, pid=os.getpid(),
        gpu_uuid=frozen.A100_UUID, checkpoint_sha256=frozen.sha256(args.checkpoint),
        manifest_sha256=frozen.sha256(args.manifest),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256={p.name:frozen.sha256(p) for p in (Path(__file__),Path(controls.__file__),Path(frozen.__file__),Path(telemetry.__file__))}, warmup_calls=WARMUP_CALLS,
        capture_attempts=1, duration_s=30, fixed_steady_window_s=[5,30],
        meaning="Native full-forward graph replay throughput on fixed resident TEST inputs; no new SRCC, not original synchronous first-block cost")
    frozen.write_json(args.output / "status.json", identity)
    timer, handles, sampler = None, [], None
    try:
        with torch.inference_mode():
            model, head, rows, prepare, transfer, forward, timer, handles = frozen.load_runtime(args)
            if any(p.requires_grad for m in (model, head) for p in m.parameters()):
                raise RuntimeError("Frozen inference only")
            cpu, positions = prepare(rows[:args.batch])
            from .diagnose_batch_numerics import input_records
            fingerprints = input_records(cpu, args.batch)
            inputs = transfer(cpu)
            reference, _, _ = forward(inputs)
            compute = controls.make_gpu_forward(args, model, head, forward)
            timer.close()
            check = compute(inputs).detach().cpu().tolist()
            if max(abs(a-b) for a,b in zip(reference,check))>.0002:
                raise RuntimeError("Native GPU-output control changed MOS")
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(WARMUP_CALLS):
                    output = compute(inputs)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            frozen.write_json(args.output / "status.json", {**identity,"status":"capturing"})
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = compute(inputs)
            graph.replay()
            torch.cuda.synchronize()
            captured = output.detach().cpu().tolist()
            delta = max(abs(a-b) for a,b in zip(reference,captured))
            if delta>.0002 or not all(__import__('math').isfinite(v) for v in captured):
                raise RuntimeError(f"Captured output changed: {delta}")
            frozen.process_audit()
            sampler = telemetry.TelemetrySampler(frozen.A100_UUID)
            sampler.start()
            sampler.set_phase("settling")
            time.sleep(2)
            sampler.set_phase("graph_replay")
            started = time.monotonic()
            groups = []
            while time.monotonic()-started<30:
                tick = time.monotonic()
                for _ in range(REPLAY_GROUP):
                    graph.replay()
                torch.cuda.synchronize()
                groups.append(dict(start_s=tick-started,end_s=time.monotonic()-started,calls=REPLAY_GROUP))
            ended = time.monotonic()
            sampler.set_phase(None)
            samples = sampler.stop();sampler=None
            telemetry.save_samples(args.output / "telemetry.csv",samples)
            frozen.write_json(args.output / "calls.json",groups)
            calls = len(groups)*REPLAY_GROUP
            full = controls.window_summary(samples,started,ended-started)
            steady = controls.window_summary(samples,started,30,5)
            time16 = (ended-started)*16000/(calls*args.batch)
            # Outside timing, verify a second, distinct batch through the same
            # graph and static addresses. Replays must not be cached predictions.
            other_cpu, other_positions = prepare(rows[args.batch:2*args.batch])
            other_hashes = input_records(other_cpu,args.batch)
            other_inputs = transfer(other_cpu)
            other_native = compute(other_inputs).detach().cpu().tolist()
            if torch.is_tensor(inputs):
                inputs.copy_(other_inputs)
            else:
                for key in inputs.keys():
                    if torch.is_tensor(inputs[key]):
                        inputs[key].copy_(other_inputs[key])
            graph.replay();torch.cuda.synchronize()
            other_graph = output.detach().cpu().tolist()
            other_delta = max(abs(a-b) for a,b in zip(other_native,other_graph))
            if other_delta>.0002:
                raise RuntimeError(f"New-input graph replay changed MOS: {other_delta}")
            frozen.write_json(args.output / "report.json", {**identity,"status":"complete",
                "sample_ids":[r['sample_id'] for r in rows[:args.batch]],"input_fingerprints":fingerprints,
                "frame_positions":positions,"started_monotonic_s":started,"ended_monotonic_s":ended,
                "calls":calls,"duration_actual_s":ended-started,"full_phase":full,"steady_5_30s":steady,
                "full_forward_time16_ms":time16,
                "energy16_j":(338.2+full['board_power_w']['mean'])*time16/1000,
                "native_mos":reference,"graph_mos":captured,"max_abs_mos_delta":delta,
                "second_batch_validation":{"sample_ids":[r['sample_id'] for r in rows[args.batch:2*args.batch]],
                    "input_fingerprints":other_hashes,"frame_positions":other_positions,
                    "native_mos":other_native,"graph_mos":other_graph,"max_abs_delta":other_delta,
                    "outside_timing":True},
                "historical_anchor_replaced":False,"process_audit_end":frozen.process_audit()})
            frozen.write_json(args.output / "status.json", {**identity,"status":"complete"})
    except BaseException as error:
        frozen.write_json(args.output / "status.json",{**identity,"status":"unsupported_or_failed","error":repr(error)})
        raise
    finally:
        if sampler is not None:
            telemetry.save_samples(args.output / "partial_telemetry.csv",sampler.stop())
        if timer is not None:
            timer.close()
        for handle in handles:
            handle.remove()


def suite(args):
    import fcntl
    reproduction.require_new_output(args.output)
    with (controls.TASK / "runs/simulation/.temporal_telemetry_a100.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        reproduction.cuda_guard()
        args.output.mkdir(parents=True,exist_ok=False)
        state=dict(status="running",completed=[],failures=[],pid=os.getpid())
        for baseline,batch in CASES:
            reproduction.cuda_guard()
            remaining=args.deadline-time.time()
            if remaining<180:
                state['status']='deadline_partial';break
            command=controls.prior.child_command(baseline,batch,args.manifest,args.output/f'{baseline}_batch{batch:02d}')
            command[command.index(controls.prior.__package__+'.evaluate_temporal_baseline_batches')]=__package__+'.audit_utilization_cuda_graph'
            command.remove('--telemetry');command.extend(['--phase','gpu'])
            name=f'{baseline}_batch{batch:02d}'
            state['phase']=name
            frozen.write_json(args.output/'suite_status.json',state)
            frozen.write_json(args.output/f'{name}.command.json',command)
            with (args.output/f'{name}.log').open('x') as stream:
                try:
                    subprocess.run(command,cwd=controls.REPO,check=True,stdout=stream,stderr=subprocess.STDOUT,timeout=min(300,remaining))
                    state['completed'].append(name)
                except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
                    state['failures'].append(dict(case=name,error=repr(error)))
            frozen.write_json(args.output/'suite_status.json',state)
        else:
            state['status']='complete_with_unsupported_cases' if state['failures'] else 'complete'
        frozen.write_json(args.output/'suite_status.json',state)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('suite','gpu'),required=True)
    parser.add_argument('--baseline',choices=('qwen','clip','yolo'))
    parser.add_argument('--batch',type=int,choices=(3,8,10))
    parser.add_argument('--deadline',type=float)
    for name in ('manifest','model','checkpoint','output','reproduction-receipt'):
        parser.add_argument('--'+name,type=Path,required=name in ('manifest','output'))
    args=parser.parse_args()
    for key,value in vars(args).items():
        if isinstance(value,Path):setattr(args,key,value.resolve())
    reproduction.require_new_output(args.output)
    if args.phase=='gpu':run(args)
    else:
        if args.deadline is None or not 0<args.deadline-time.time()<=7200:raise ValueError('Bounded deadline required')
        suite(args)


if __name__=='__main__':main()
