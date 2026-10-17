"""Finite native reproduction at B3/B10, with only B4/B11 conditional steps.

The native cost/resident worker and complete-window aggregation are reused as-is.
New process per phase/point, no thermal conditioning or measurement-window change.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from . import measure_frozen_batch_costs as costs
from . import unified_resident_utilization as uniform
from . import evaluate_temporal_baseline_batches as frozen
from . import reproduce_temporal_baseline as reproduction

TASK, REPO = costs.TASK, costs.REPO
CONFIG = TASK/'configs/baselines/yolo_clip_recheck_20261017.json'


def config(profile='fresh'):
    file=CONFIG if profile=='fresh' else TASK/'configs/baselines/yolo_clip_sequential_20261017.json'
    c=json.loads(file.read_text())
    expected={'yolo':[3,4],'clip':[10,11]} if profile=='fresh' else {'yolo':[1,2,3],'clip':[1,2,3,4,6,8,9,10]}
    if c['cases']!=expected:raise ValueError('Only the prescribed finite batch plan')
    u=uniform.config()
    if any(c[k]!=u[k] for k in ['warmup_s','measurement_s','repeats','queue_depth']):raise ValueError('Original measurement loop contract')
    if c['target_percent']!=98 or c['maximum_suite_seconds']!=3600:raise ValueError('Finite original target')
    return c


def needs_extra(mean):
    return float(mean)<config()['target_percent']


def sources():
    return {**costs.sources(),**{p.relative_to(REPO).as_posix():frozen.sha256(p) for p in [Path(__file__),CONFIG,TASK/'configs/baselines/yolo_clip_sequential_20261017.json']}}


def sequential_protocol(model):
    c=config('sequential');protocol=uniform.config()
    protocol['batches']={model:c['cases'][model]}
    protocol['required_terminal_batch']=c['target_batches'][model]
    protocol['sequential_reproduction']=c
    return protocol


def worker(args):
    c=config(args.profile);bound=sources()
    if args.profile=='sequential':
        if args.phase!='util' or args.baseline not in c['cases']:raise ValueError('Sequential native utilization only')
        uniform.worker(args,sequential_protocol(args.baseline))
        if bound!=sources() or frozen.sha256(args.checkpoint)!=costs.CHECKPOINTS[args.baseline]:raise RuntimeError('Source/PT changed')
        result=json.loads((args.output/'report.json').read_text())
        if [p['batch'] for p in result['points']]!=c['cases'][args.baseline]:raise RuntimeError('Required ascending plan incomplete')
        frozen.write_json(args.output/'sequential_driver_receipt.json',dict(status='complete',baseline=args.baseline,protocol=c,source_sha256=bound,checkpoint_sha256=frozen.sha256(args.checkpoint)))
        return
    if args.baseline not in c['cases'] or args.batch not in c['cases'][args.baseline]:raise ValueError('Unsupported point')
    if args.phase=='cost':
        protocol=costs.config();protocol['batches']=c['cases'];protocol['new_authorization']=c
        costs.worker(args,protocol)
    else:
        protocol=uniform.config();protocol['batches']={args.baseline:[args.batch]}
        uniform.worker(args,protocol)
    if bound!=sources() or frozen.sha256(args.checkpoint)!=costs.CHECKPOINTS[args.baseline]:raise RuntimeError('Source/PT changed')
    frozen.write_json(args.output/'recheck_driver_receipt.json',dict(status='complete',baseline=args.baseline,batch=args.batch,phase=args.phase,protocol=c,source_sha256=bound,checkpoint_sha256=frozen.sha256(args.checkpoint)))


def suite(args):
    import fcntl,signal
    c=config(args.profile);reproduction.require_new_output(args.output)
    with (TASK/'runs/simulation/.temporal_telemetry_a100.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        costs.wait_for_idle(args,c);args.output.mkdir()
        state=dict(status='running',pid=os.getpid(),protocol=c,source_sha256=sources(),completed=[],skipped=[],deadline=args.deadline,started_unix=time.time())
        def save():frozen.write_json(args.output/'suite_status.json',state)
        save()
        try:
            for model,batches in c['cases'].items():
                if args.profile=='sequential':
                    if time.time()>=args.deadline or sources()!=state['source_sha256']:raise RuntimeError('Original deadline/source')
                    reproduction.cuda_guard();folder=args.output/model
                    command=costs.child_command(model,c['target_batches'][model],args,folder)
                    command[command.index(__package__+'.measure_frozen_batch_costs')]=__package__+'.measure_yolo_clip_recheck'
                    command[command.index('--phase')+1]='util';command+=['--profile','sequential']
                    state.update(phase=model,phase_started_unix=time.time());save()
                    frozen.write_json(args.output/(model+'.command.json'),command)
                    with (args.output/(model+'.log')).open('x') as stream:
                        p=subprocess.Popen(command,cwd=REPO,env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                        try:code=p.wait(timeout=min(c['maximum_stage_seconds'],max(1,args.deadline-time.time())))
                        except subprocess.TimeoutExpired:
                            os.killpg(p.pid,signal.SIGTERM)
                            try:p.wait(timeout=10)
                            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
                            raise
                    if code:raise RuntimeError(f'{model} exited {code}; no automatic retry')
                    if json.loads((folder/'report.json').read_text())['status']!='complete':raise RuntimeError('Incomplete model plan')
                    state['completed'].append(model);save();continue
                for index,batch in enumerate(batches):
                    if time.time()>=args.deadline or sources()!=state['source_sha256']:raise RuntimeError('Original deadline/source')
                    folder=args.output/f'{model}_batch{batch:02d}';folder.mkdir()
                    for phase in ['cost','util']:
                        reproduction.cuda_guard()
                        command=costs.child_command(model,batch,args,folder/phase)
                        command[command.index(__package__+'.measure_frozen_batch_costs')]=__package__+'.measure_yolo_clip_recheck'
                        command[command.index('--phase')+1]=phase
                        state.update(phase=folder.name+'/'+phase,phase_started_unix=time.time());save()
                        frozen.write_json(folder/(phase+'.command.json'),command)
                        with (folder/(phase+'.log')).open('x') as stream:
                            p=subprocess.Popen(command,cwd=REPO,env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                            try:code=p.wait(timeout=min(c['maximum_stage_seconds'],max(1,args.deadline-time.time())))
                            except subprocess.TimeoutExpired:
                                os.killpg(p.pid,signal.SIGTERM)
                                try:p.wait(timeout=10)
                                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
                                raise
                        if code:raise RuntimeError(f'{folder.name}/{phase} exited {code}; no automatic retry')
                        r=json.loads((folder/phase/'report.json').read_text())
                        if r['status']!='complete':raise RuntimeError('Incomplete worker')
                    state['completed'].append(folder.name);save()
                    result=json.loads((folder/'util/report.json').read_text())
                    if len(result['points'])!=1:raise RuntimeError('One independent point per process')
                    mean=result['points'][0]['mean']
                    if index==0 and not needs_extra(mean):
                        state['skipped'].append(dict(model=model,batch=batches[1],reason='first prescribed point pooled mean >=98'));save();break
            reproduction.cuda_guard();state.update(status='complete',completed_unix=time.time());save()
        except BaseException as error:
            state.update(status='failed',error=repr(error));save();raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['suite','cost','util'],required=True)
    p.add_argument('--profile',choices=['fresh','sequential'],default='fresh')
    p.add_argument('--baseline',choices=['yolo','clip']);p.add_argument('--batch',type=int)
    for name in ['manifest','output','model','checkpoint','reproduction-receipt']:p.add_argument('--'+name,type=Path,required=name in ['manifest','output'])
    p.add_argument('--deadline',type=float,required=True);args=p.parse_args();args.output=args.output.resolve()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=frozen.A100_UUID or frozen.sha256(args.manifest)!=frozen.MANIFEST_SHA:raise ValueError('Approved physical A100 and original split only')
    if not 0<args.deadline-time.time()<=config(args.profile)['maximum_suite_seconds']:raise ValueError('At most one hour, no extension')
    try:(suite if args.phase=='suite' else worker)(args)
    except BaseException as error:
        if args.output.exists():frozen.write_json(args.output/'failure.json',dict(error=repr(error),pid=os.getpid(),no_automatic_retry=True))
        raise


if __name__=='__main__':main()
