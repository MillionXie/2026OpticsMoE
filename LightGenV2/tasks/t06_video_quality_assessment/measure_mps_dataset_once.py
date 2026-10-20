"""Cold, ordered distinct-video MPS costs, preserving legacy block timers.

No inference before TEST sample zero. Include output-to-next-PREPROCESS-start
gaps, excluding all external preprocessing and model-call preblock frontend.
Legacy compute-only cost and continuous-wall cost remain separate audit fields.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from . import lab_busy_resource_20261020 as lab
from . import rental_busy_resource_20261020 as engine
from . import evaluate_temporal_baseline_batches as frozen
from . import measure_frozen_batch_costs as costs
from . import unified_resident_utilization as uniform
from . import batch_telemetry as telemetry
from . import strict_tf_window_20261020 as energy
from .nvml_observer_process import ProcessTelemetrySampler

TASK=Path(__file__).resolve().parent
REPO=TASK.parents[2]
CONFIG=TASK/'configs/baselines/mps_dataset_once_20261020.json'
ROOT=TASK/'runs/simulation/lgvq_mps_dataset_once_a100_20261020'
PIPE='/DATA/DATA1/guest3/.cache/t06_mps_dataset_once_20261020'
PROFILE='lab'


def select_profile(name):
    """Published machine profile; never mutate the completed laboratory run."""
    global PROFILE,CONFIG,ROOT,PIPE
    PROFILE=name
    if name.startswith('rental'):
        suffix='_replacement02' if name=='rentalb' else ''
        CONFIG=TASK/f'configs/baselines/mps_dataset_once_rental_20261021{suffix}.json'
        ROOT=TASK/f'runs/simulation/lgvq_mps_dataset_once_rental_20261021{suffix}'
        PIPE='/data/tmp/t06_mps_dataset_once_20261021'+suffix
        frozen.A100_UUID=json.loads(CONFIG.read_text())['gpu_uuid']


def config():
    c=json.loads(CONFIG.read_text())
    assert c['models']==(['deepseek','kimi'] if PROFILE.startswith('rental') else ['qwen','glm','clip','yolo']) and c['resource_caps']==[100,50]
    assert (c['batch'],c['test_videos'],c['cost_videos'],c['repeats'],c['explicit_warmup_forwards'])==(1,558,200,3,0)
    return c


def sources():
    return {**lab.sources(),**{str(p.relative_to(REPO)):frozen.sha256(p)
                              for p in (Path(__file__),CONFIG)}}


def charged_intervals(calls):
    """Nonpreprocessing gaps end BEFORE next decode/processor/H2D starts."""
    if not calls:raise ValueError('No calls')
    intervals=[];compute=gap=0.
    for i,c in enumerate(calls):
        p,a,s,e=(float(c[k]) for k in ('preprocess_start','model_call_start','first_block_start','gpu_output'))
        if not p<=a<=s<e:raise ValueError('Invalid ordered boundaries')
        intervals.append((s,e));compute+=e-s
        if i:
            previous=float(calls[i-1]['gpu_output'])
            if previous>p:raise ValueError('Overlapping calls')
            if previous<p:intervals.append((previous,p));gap+=p-previous
    return sorted(intervals),compute,gap


def write_csv(path,records):
    with path.open('x',newline='') as stream:
        if records:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)


def validate_idle_memory(gpu):
    """Invisible host allocations can exist despite an empty container PID list."""
    raw=subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    used=float(raw.strip())
    if used>1024:raise RuntimeError(f'Idle admission failed: {used:g} MiB remains allocated; require provider diagnosis, no reset/kill')
    return used


def runtime(args,guard,identity):
    """Reuse native arithmetic AND original timer, not the resident timing hook."""
    import torch
    from . import quality_token_common as core
    from LightGenV2.common.baseline_measurement import FirstBlockTimer
    a=SimpleNamespace(**vars(args));a.baseline=args.model;a.model=args.model_path
    if args.model in ('deepseek','kimi'):
        import inspect
        # Reuse the receipt-bound native arithmetic, without any smoke/warmup.
        a.model=args.model;a.precision='native';a.dual_fp32=False;a.gpu_uuid=frozen.A100_UUID
        defer=SimpleNamespace(bind=lambda:None)
        model,processor,prompt,first,transfer,compute=engine.base.runtime(a,defer,identity)
        head=inspect.getclosurevars(compute).nonlocals['head']
        rows=[r for r in engine.base.old.rows() if r['split']=='test']
        prepare=lambda selected:engine.base.prior.prepare_batch(args.model,selected,processor,prompt)
        timer=FirstBlockTimer(first)
        def forward(inputs):
            timer.reset();output,native=compute(inputs);timing=timer.finish()
            return output.detach().cpu().tolist(),timing['host_ms'],timing['cuda_ms']
        def start():return timer._state['host_started']
        assets=identity['assets']
        def finish_all():
            timer.close()
            if engine.base.old.verify_assets(Path('/data/models')/args.model)!=assets:raise RuntimeError('Assets changed')
    elif args.model=='glm':
        model,head,prepare,transfer,compute,_,finish=uniform.special_runtime(a)
        rows=[r for r in core.read_manifest(args.manifest) if r['split']=='test']
        timer=FirstBlockTimer(model.model.visual.blocks[0])
        def forward(inputs):
            torch.cuda.synchronize();timer.reset();output=compute(inputs);timing=timer.finish()
            return output.detach().cpu().tolist(),timing['host_ms'],timing['cuda_ms']
        def start():return timer._state['host_started']
        assets=costs.model_hashes(args.model_path)
        def finish_all():timer.close();finish()
    else:
        assets=costs.model_hashes(args.model_path)
        model,head,rows,prepare,transfer,forward,timer,handles=frozen.load_runtime(a)
        def start():return timer.state['vision_host_started'] if args.model=='qwen' else timer._state['host_started']
        def finish_all():
            timer.close()
            for handle in handles:handle.remove()
            if costs.model_hashes(args.model_path)!=assets:raise RuntimeError('Assets changed')
    if len(rows)!=558 or len({r['sample_id'] for r in rows})!=558:raise RuntimeError('Original558 unique TEST required')
    expected=identity['checkpoint_sha256'] if PROFILE.startswith('rental') else costs.CHECKPOINTS[args.model]
    if frozen.sha256(args.checkpoint)!=expected:raise RuntimeError('Exact frozen head differs')
    if any(p.requires_grad for m in (model,head) for p in m.parameters()):raise RuntimeError('Weights must be frozen')
    identity.update(assets=assets,checkpoint_sha256=frozen.sha256(args.checkpoint),
                    floating_parameter_dtypes=sorted({str(p.dtype) for p in model.parameters() if p.is_floating_point()}),
                    quality_readout_dtypes=sorted({str(p.dtype) for p in head.parameters() if p.is_floating_point()}))
    guard.bind()
    return rows,prepare,transfer,forward,start,finish_all


def worker(args):
    import torch
    import numpy as np
    from . import quality_token_common as core
    c=config();engine.GPU=frozen.A100_UUID;engine.PIPE=PIPE;engine.ROOT=ROOT
    if args.output.exists():raise RuntimeError('Preserve prior worker evidence')
    if args.model not in ('glm','deepseek','kimi'):
        pre=SimpleNamespace(**vars(args));pre.baseline=args.model;pre.model=args.model_path
        frozen.preflight_files(pre)
    if os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE')!=str(args.cap):raise RuntimeError('Set MPS cap before CUDA')
    args.output.mkdir();bound=sources()
    class Guard(engine.MPSGuard):
        def check(self):
            if sources()!=bound:raise RuntimeError('Bound source changed')
            super().check()
    guard=Guard(args);sampler=None;finish=None;calls=[];predictions=[]
    identity=dict(model=args.model,cap=args.cap,repeat=args.repeat,batch=1,precision='original_deployment',
                  protocol=c,pid=os.getpid(),command=sys.argv,gpu_uuid=frozen.A100_UUID,
                  git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),sources=bound,
                  manifest_sha256=frozen.sha256(args.manifest),explicit_warmup_forwards=0,
                  first_test_batch_included=True,clock_offset_s=time.perf_counter()-time.monotonic())
    identity['timer_start_placement']='legacy first-block host entry; no extra in-hook preblock drain'
    def status(value,**extra):frozen.write_json(args.output/'status.json',dict(status=value,time=time.time(),pid=os.getpid(),**extra))
    try:
        guard.check();torch.set_num_threads(8)
        if abs(identity['clock_offset_s'])>1e-4:raise RuntimeError('Legacy perf and observer monotonic clocks differ')
        if identity['manifest_sha256']!=frozen.MANIFEST_SHA:raise RuntimeError('Original split SHA required')
        limit=float(subprocess.check_output(['nvidia-smi','--id='+frozen.A100_UUID,'--query-gpu=power.limit','--format=csv,noheader,nounits'],text=True))
        if abs(limit-c['power_limit_w'])>.1:raise RuntimeError('Do not change original platform power cap')
        status('loading')
        with torch.inference_mode():
            rows,prepare,transfer,forward,start,finish=runtime(args,guard,identity)
            visible=torch.cuda.get_device_properties(0).multi_processor_count
            if visible!=(108 if args.cap==100 else 54):raise RuntimeError('MPS allocation not effective')
            identity.update(visible_multiprocessors=visible,head_and_backbone_frozen=True,
                            gpu_name=torch.cuda.get_device_name(0),power_limit_w=limit,execution_profile=PROFILE)
            frozen.write_json(args.output/'identity.json',identity)
            sampler=ProcessTelemetrySampler(frozen.A100_UUID,args.output/'telemetry_stdout.jsonl')
            sampler.start();sampler.set_phase('dataset_once_no_warmup')
            # CPU-only power settling; absolutely no forward or synthetic GPU load.
            time.sleep(.3)
            with (args.output/'predictions.csv').open('x',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(['sample_index','sample_id','target_mos','prediction','frame_positions'])
                for index,row in enumerate(rows):
                    guard.fast();sampler.check()
                    p=time.perf_counter();cpu,positions=prepare([row]);decoded=time.perf_counter()
                    inputs=transfer(cpu);torch.cuda.synchronize();a=time.perf_counter()
                    values,wall,cuda=forward(inputs);returned=time.perf_counter();s=start();e=s+wall/1000
                    if len(values)!=1 or not math.isfinite(values[0]) or not p<=a<=s<e<=returned+1e-5:
                        raise RuntimeError('Invalid prediction or legacy timing boundary')
                    calls.append(dict(sample_index=index,sample_id=row['sample_id'],preprocess_start=p,
                                      external_preprocess_end=a,model_call_start=a,first_block_start=s,gpu_output=e,
                                      call_return=returned,decode_processor_s=decoded-p,h2d_sync_s=a-decoded,
                                      excluded_model_preblock_s=s-a,legacy_host_ms=wall,cuda_ms=cuda,
                                      frame_positions=json.dumps(positions[0]),in_cost_subset=index<200))
                    predictions.append(float(values[0]));writer.writerow([index,row['sample_id'],row['temporal'],values[0],json.dumps(positions[0])])
                    del inputs,cpu
                    if index==0 or (index+1)%25==0 or index==557:
                        stream.flush();status('evaluating',samples=index+1,total=558)
                        print(f'{args.model} cap{args.cap} repeat{args.repeat} {index+1}/558 wall={wall:.3f}ms',flush=True)
            sampler.request('barrier');samples=sampler.stop();sampler=None
            telemetry.save_samples(args.output/'telemetry.csv',samples)
            intervals,compute,gap=charged_intervals(calls[:200]);result=energy.interval_energy(intervals,samples,c['host_proxy_w'])
            if not math.isclose(result['elapsed_s'],compute+gap,abs_tol=1e-7):raise RuntimeError('Time union mismatch')
            raw=[s for s in samples if calls[0]['first_block_start']<=s.host_monotonic_s<=calls[199]['gpu_output']]
            report=dict(status='complete',**identity,test_videos=558,cost_videos=200,
                        first200_unique_videos=len({r['sample_id'] for r in calls[:200]}),
                        performance=core.metrics(np.array([float(r['temporal']) for r in rows]),np.array(predictions)),
                        cost=dict(time16_ms=result['elapsed_s']*80,energy16_j=result['energy_j']*.08,
                                  charged_s=result['elapsed_s'],compute_s=compute,nonpreprocessing_gap_s=gap,
                                  legacy_compute_only_time16_ms=compute*80,estimated_energy_j=result['energy_j'],
                                  excluded_external_preprocess_s=sum(r['external_preprocess_end']-r['preprocess_start'] for r in calls[:200]),
                                  excluded_model_preblock_s=sum(r['excluded_model_preblock_s'] for r in calls[:200]),
                                  power_integration=result),
                        full_wall_telemetry=telemetry.telemetry_report(raw,prefix=''),
                        accuracy_selection='Frozen existing head; TEST development metric, noVAL; no model selection',
                        historical_Qwen_B1_unchanged=True)
            finish();finish=None;guard.check();frozen.write_json(args.output/'report.json',report);status('complete',samples=558)
    except BaseException as error:status('failed',error=repr(error),samples=len(predictions));raise
    finally:
        if sampler is not None:telemetry.save_samples(args.output/'telemetry.csv',sampler.stop(raise_on_error=False))
        if finish is not None:finish()
        write_csv(args.output/'calls.csv',calls);frozen.write_json(args.output/'identity.json',identity)
        guard.close(args.output/'process_audit.json')


def suite(args):
    import fcntl
    with (TASK/'runs/simulation/.temporal_telemetry_a100.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if ROOT.exists():raise RuntimeError('Do not overwrite suite')
        ROOT.mkdir();state=dict(status='admission',pid=os.getpid(),completed=[],deadline=args.deadline,protocol=config())
        def save():frozen.write_json(ROOT/'suite_status.json',state)
        save();engine.GPU=frozen.A100_UUID;engine.PIPE=PIPE;engine.ROOT=ROOT;active=False
        try:
            # Bounded consecutive idle checks, never occupy or stop other users.
            for _ in range(3):
                if lab.shared.guard_records(frozen.A100_UUID):raise RuntimeError('A100 is no longer idle')
                validate_idle_memory(frozen.A100_UUID)
                if time.time()>=args.deadline:raise TimeoutError('Suite deadline')
                time.sleep(10)
            engine.daemon(True);active=True
            for model in config()['models']:
                if PROFILE.startswith('rental'):
                    mp=Path('/data/models')/model
                    pt=engine.base.old.RUN/model/'train/best_checkpoint.pt'
                    python=f'/data/venvs/{model}/bin/python';receipt=None
                else:mp,pt,python,receipt=lab.locations(model)
                for cap in (100,50):
                    for repeat in (1,2,3):
                        if time.time()>=args.deadline-120:raise TimeoutError('No time for another worker')
                        name=f'{model}_cap{cap}_repeat{repeat:02d}';output=ROOT/name
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES=frozen.A100_UUID,CUDA_MPS_PIPE_DIRECTORY=PIPE,
                                 CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=str(cap),OMP_NUM_THREADS='8',
                                 HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME='/data/cache')
                        command=[python,'-u','-m',__package__+'.measure_mps_dataset_once','--phase','worker',
                                 '--model',model,'--cap',str(cap),'--repeat',str(repeat),'--model-path',str(mp),
                                 '--checkpoint',str(pt),'--manifest',str(args.manifest),'--output',str(output),'--deadline',str(args.deadline),
                                 '--profile',PROFILE]
                        if receipt:command.extend(['--reproduction-receipt',str(receipt)])
                        with (ROOT/(name+'.log')).open('x') as log:
                            child=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                            state.update(status='running',phase=name,child_pid=child.pid);save()
                            try:code=child.wait(timeout=min(3600,max(1,args.deadline-time.time())))
                            except BaseException:
                                os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=15);raise
                        if code:raise RuntimeError(f'{name} stopped exit{code}; preserve, no automatic retry')
                        state['completed'].append(name);save()
            state.update(status='complete',finished=time.time())
        except BaseException as error:state.update(status='stopped',error=repr(error));raise
        finally:
            if active:
                try:engine.daemon(False);state['mps_released']=True
                except BaseException as error:state['release_error']=repr(error)
            save()


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('suite','worker'),required=True)
    p.add_argument('--profile',choices=('lab','rental','rentalb'),default='lab')
    p.add_argument('--model',choices=('qwen','glm','clip','yolo','deepseek','kimi'));p.add_argument('--cap',type=int,choices=(100,50))
    p.add_argument('--repeat',type=int,choices=(1,2,3));p.add_argument('--model-path',type=Path)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--reproduction-receipt',type=Path)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path)
    p.add_argument('--deadline',type=float,required=True);a=p.parse_args();select_profile(a.profile)
    if a.phase=='suite':suite(a)
    else:worker(a)


if __name__=='__main__':main()
