"""Bounded Qwen metadata/native and CLIP native traces; no formal data writes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from . import measure_frozen_batch_costs as costs
from . import evaluate_temporal_baseline_batches as frozen
from . import run_temporal_telemetry_suite as prior
from . import reproduce_temporal_baseline as reproduction
from . import unified_resident_utilization as uniform
from . import extend_clip_diagnose_qwen as previous
from . import measure_qwen_host_metadata as metadata

TASK, REPO = prior.TASK, prior.REPO
CONFIG = TASK/'configs/baselines/residual_sync_20261016.json'


def config():
    c=json.loads(CONFIG.read_text())
    if c['cases']!={'qwen':[2,4],'clip':[4,8,10]} or not c['no_B1_execution_or_replacement']:
        raise ValueError('Finite small-batch diagnosis, no B1')
    if c['profile_forward_calls']!=4 or c['maximum_suite_seconds']!=3600:
        raise ValueError('Bounded profile budget')
    return c


def sources():
    return {**metadata.sources(),str(Path(__file__).relative_to(REPO)):frozen.sha256(Path(__file__)),
            str(CONFIG.relative_to(REPO)):frozen.sha256(CONFIG),
            str((TASK/'measure_residual_sync.py').relative_to(REPO)):frozen.sha256(TASK/'measure_residual_sync.py'),
            str((TASK/'residual_sync_execution.py').relative_to(REPO)):frozen.sha256(TASK/'residual_sync_execution.py'),
            str(Path(previous.__file__).relative_to(REPO)):frozen.sha256(Path(previous.__file__))}


def worker(args):
    frozen.preflight_files(args);reproduction.cuda_guard()
    import torch
    from . import audit_utilization_execution as execution
    from . import quality_token_common as core
    from .qwen_host_metadata_execution import HostMetadataQwenModel
    from LightGenV2.common.baseline_measurement import gpu_power_limit_w
    if abs(gpu_power_limit_w()-250)>.1:raise RuntimeError('Original250W required')
    args.output.mkdir()
    bound=sources();assets=costs.model_hashes(args.model)
    identity=dict(source_sha256=bound,asset_sha256=assets,checkpoint_sha256=frozen.sha256(args.checkpoint),
        manifest_sha256=frozen.sha256(args.manifest),protocol=config(),pid=os.getpid(),command=sys.argv,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
    frozen.write_json(args.output/'identity.json',identity)
    result=dict(status='running',baseline=args.baseline,points=[],profiler_is_not_formal_utilization=True)
    timer=None;handles=[]
    try:
        with torch.inference_mode():
            model,head,_,prepare,transfer,original,timer,handles=frozen.load_runtime(args)
            if any(p.requires_grad for m in (model,head) for p in m.parameters()):raise RuntimeError('Frozen only')
            compute=execution.make_gpu_forward(args,model,head,original)
            rows=[r for r in core.read_manifest(args.manifest) if r['split']=='test']
            if len(rows)!=558:raise RuntimeError('Original TEST required')
            timer_closed=False
            for batch in ([2,3] if args.candidate_execution else config()['cases'][args.baseline]):
                if time.time()>=args.deadline or sources()!=bound:raise RuntimeError('Deadline/source guard')
                folder=args.output/f'batch{batch:02d}';folder.mkdir()
                cpu,positions=prepare(rows[:batch]);inputs=transfer(cpu)
                if not timer_closed:
                    reference=torch.tensor(original(inputs)[0]);timer.close();timer_closed=True
                else:reference=compute(inputs).detach().cpu()
                if not torch.equal(reference,compute(inputs).detach().cpu()):raise RuntimeError('Original native MOS changed')
                modes=[('native',compute)]
                if args.candidate_execution:
                    from .residual_sync_execution import IndexedHostQwen
                    from . import measure_residual_sync as strict
                    adapter=IndexedHostQwen(model.model,cpu,inputs)
                    candidate=execution.make_gpu_forward(args,SimpleNamespace(model=adapter,lm_head=model.lm_head),head,original)
                    with metadata.validation_guard(folder/'equality_process_audit.json'):
                        eq,_,_=strict.strict_pair(compute,candidate,inputs,model.model,adapter,model,head,'qwen',True)
                        frozen.write_json(folder/'strict_equality.json',eq)
                    modes=[('indexed_candidate',candidate)]
                elif args.baseline=='qwen':
                    adapter=HostMetadataQwenModel(model.model,cpu,inputs)
                    candidate=execution.make_gpu_forward(args,SimpleNamespace(model=adapter,lm_head=model.lm_head),head,original)
                    with metadata.validation_guard(folder/'equality_process_audit.json'):
                        eq,_,_=metadata.strict_pair(compute,candidate,inputs,model,head)
                        frozen.write_json(folder/'strict_equality.json',eq)
                    modes.append(('host_metadata',candidate))
                point=dict(batch=batch,modes={},sample_ids=[r['sample_id'] for r in rows[:batch]],frame_positions=positions)
                for mode,fn in modes:
                    frozen.write_json(args.output/'status.json',dict(status='profiling',batch=batch,mode=mode,updated_unix=time.time()))
                    with metadata.validation_guard(folder/(mode+'_process_audit.json')) as guard:
                        # Same forward, warm outside the diagnostic trace; not a utilization measurement.
                        start=time.monotonic()
                        while time.monotonic()-start<5:
                            for _ in range(8):fn(inputs)
                            torch.cuda.synchronize();guard.check()
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],with_stack=True) as prof:
                            for _ in range(config()['profile_forward_calls']):
                                with torch.profiler.record_function(mode+'_resident_forward'):
                                    output=fn(inputs)
                            torch.cuda.synchronize()
                    if not torch.equal(output.detach().cpu(),reference):raise RuntimeError('Profiled MOS changed')
                    target=folder/(mode+'_trace.json');prof.export_chrome_trace(str(target))
                    if target.stat().st_size>config()['profile_maximum_bytes']:raise RuntimeError('Trace limit; preserve evidence')
                    summary=previous.trace_summary(json.loads(target.read_text()))
                    operators=[dict(name=e.key,calls=e.count,self_cpu_us=e.self_cpu_time_total,inclusive_cpu_us=e.cpu_time_total,
                        stack=list(e.stack)) for e in prof.key_averages(group_by_stack_n=5)]
                    point['modes'][mode]=dict(profiled=summary,operators=sorted(operators,key=lambda x:-x['self_cpu_us'])[:80],
                        scalar_or_nonzero=[x for x in operators if any(k in x['name'] for k in ['item','nonzero','_local_scalar_dense'])],
                        native_mos=reference.tolist(),profile_mos=output.detach().cpu().tolist(),max_abs=0)
                    frozen.write_json(folder/(mode+'_summary.json'),point['modes'][mode])
                    print(json.dumps(dict(model=args.baseline,batch=batch,mode=mode,kernel_union=summary['kernel_union_fraction'])),flush=True)
                    del prof
                frozen.write_json(folder/'report.json',point);result['points'].append(point)
                frozen.write_json(args.output/'report.json',result)
                del inputs,cpu
            if sources()!=bound or costs.model_hashes(args.model)!=assets:raise RuntimeError('Bound files changed')
            result.update(status='complete',completed_unix=time.time());frozen.write_json(args.output/'report.json',result)
    except BaseException as error:
        result.update(status='failed',error=repr(error),no_automatic_retry=True);frozen.write_json(args.output/'report.json',result);raise
    finally:
        if timer is not None:timer.close()
        for h in handles:h.remove()


def suite(args):
    import fcntl
    reproduction.require_new_output(args.output);args.output.mkdir()
    bound=sources();state=dict(status='waiting_for_idle',pid=os.getpid(),deadline=args.deadline,completed=[],source_sha256=bound,protocol=config())
    def save():frozen.write_json(args.output/'suite_status.json',state)
    save()
    with (TASK/'runs/simulation/.temporal_telemetry_a100.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            costs.wait_for_idle(args,config())
            for model in (['qwen'] if args.candidate_execution else config()['cases']):
                reproduction.cuda_guard()
                if sources()!=bound:raise RuntimeError('Source changed')
                state.update(status='running',model=model,updated_unix=time.time());save()
                cmd=prior.child_command(model,config()['cases'][model][0],args.manifest,args.output/model)
                cmd[cmd.index(__package__+'.evaluate_temporal_baseline_batches')]=__package__+'.diagnose_residual_sync'
                cmd.remove('--telemetry');cmd+=['--phase','worker','--deadline',str(args.deadline)]
                if args.candidate_execution:cmd+=['--candidate-execution']
                frozen.write_json(args.output/(model+'.command.json'),cmd)
                with (args.output/(model+'.log')).open('x') as stream:
                    subprocess.run(cmd,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=min(config()['maximum_worker_seconds'],args.deadline-time.time()))
                if json.loads((args.output/model/'report.json').read_text())['status']!='complete':raise RuntimeError('Incomplete worker')
                state['completed'].append(model);save()
            state.update(status='complete',completed_unix=time.time());save()
        except BaseException as error:
            state.update(status='failed',error=repr(error),no_automatic_retry=True);save();raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['suite','worker'],required=True)
    p.add_argument('--baseline',choices=['qwen','clip']);p.add_argument('--batch',type=int,default=2)
    p.add_argument('--candidate-execution',action='store_true')
    for name in ['manifest','output','model','checkpoint','reproduction-receipt']:
        p.add_argument('--'+name,type=Path,required=name in ['manifest','output'])
    p.add_argument('--deadline',type=float,required=True)
    args=p.parse_args();args.output=args.output.resolve()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=frozen.A100_UUID or frozen.sha256(args.manifest)!=frozen.MANIFEST_SHA:
        raise RuntimeError('Original A100/manifest required')
    if not 0<args.deadline-time.time()<=config()['maximum_suite_seconds']:raise ValueError('Bounded one-hour trial')
    if args.phase=='suite':suite(args)
    else:
        if args.batch not in config()['cases'][args.baseline] or frozen.sha256(args.checkpoint)!=costs.CHECKPOINTS[args.baseline]:raise RuntimeError('Frozen model/batch contract')
        worker(args)


if __name__=='__main__':main()
