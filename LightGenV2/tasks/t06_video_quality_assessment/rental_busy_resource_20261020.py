"""Authorized80GB paired complete-window busy-time and MPS resource experiments.

Same prepared first-B inputs; real inference on every call. No output cache,
offload, Graph, training, frequency or power change. First TF input to final GPU
MOS synchronization includes INTERNAL waits, never a wait after final output.
Equivalent16-video time is a throughput normalization, not a16-video batch.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
from . import rental_precision_recheck as base

TASK=base.TASK;REPO=base.REPO
ROOT=TASK/'runs/simulation/lgvq_busy_resource_20261020'
PIPE='/data/tmp/t06_mps_20261020'
GPU='GPU-780326a6-79bc-7a45-36f6-683c5b278bee'


def sources():return {**base.sources(),str(Path(__file__).relative_to(REPO)):base.old.sha(Path(__file__))}


def mps(command):
    result=subprocess.run(['nvidia-cuda-mps-control'],input=command+'\n',text=True,capture_output=True,timeout=5,
                          env=dict(os.environ,CUDA_MPS_PIPE_DIRECTORY=PIPE))
    if result.returncode:raise RuntimeError('MPS control failed: '+result.stderr+result.stdout)
    return result.stdout


def numeric_lines(raw):return {int(x) for x in re.findall(r'^\s*(\d+)\s*$',raw,re.M)}


class MPSGuard:
    def __init__(self,args):
        self.args=args;self.audit=[];self.bound=sources();self.error=None;self.stop=threading.Event();self.thread=None;self.bound_client=False
        line=next((s for s in Path('/proc/self/status').read_text().splitlines() if s.startswith('NSpid:')),'')
        self.self_ids={os.getpid(),*[int(x) for x in line.split()[1:]]}
    def check(self):
        if time.time()>=self.args.deadline:raise TimeoutError('Original finite window')
        if sources()!=self.bound:raise RuntimeError('Source changed')
        servers=numeric_lines(mps('get_server_list'));clients=set();controls={}
        for server in servers:
            raw=mps('get_client_list '+str(server));controls[str(server)]=raw;clients|=numeric_lines(raw)
        processes=base.guard_records(GPU)
        if clients-self.self_ids:raise RuntimeError('Foreign MPS client')
        if self.bound_client and not clients&self.self_ids:raise RuntimeError('Own MPS client disappeared')
        if {int(r[0].strip()) for r in processes}-(servers|self.self_ids):raise RuntimeError('Foreign non-MPS GPU process')
        self.audit.append(dict(unix=time.time(),servers=sorted(servers),clients=sorted(clients),processes=processes,control=controls))
    def bind(self):
        self.bound_client=True;self.check()
        def monitor():
            while not self.stop.wait(1):
                try:self.check()
                except BaseException as error:self.error=error;return
        self.thread=threading.Thread(target=monitor,daemon=True);self.thread.start()
    def fast(self):
        if self.error:raise RuntimeError('MPS independent guard failed') from self.error
        if time.time()>=self.args.deadline:raise TimeoutError('Original finite window')
    def close(self,path):
        self.stop.set()
        if self.thread:self.thread.join(6)
        base.old.write(path,self.audit)


def complete_window(args,compute,inputs,first,folder,guard,duty):
    import torch
    folder.mkdir();sampler=base.telemetry.TelemetrySampler(GPU,raw_path=folder/'telemetry_stdout.jsonl')
    clocks=base.prior.ClockSampler(GPU);repeats=[];events=[];armed=dict(start=None)
    hook=first.register_forward_pre_hook(lambda *_:armed.update(start=time.monotonic()) if armed['start'] is None else None)
    sampler.start();clocks.start()
    def group():
        guard.fast();sampler.check();start=time.monotonic()
        for _ in range(8):out,native=compute(inputs)
        torch.cuda.synchronize();end=time.monotonic()
        return out,start,end
    def idle(seconds):
        start=time.monotonic();end=start+seconds
        while time.monotonic()<end:
            guard.fast();sampler.check();time.sleep(min(.05,max(0,end-time.monotonic())))
        return start,time.monotonic()
    try:
        # Pilot determines only the bounded complete-work count, not window selection.
        warm=time.monotonic();times=[]
        while time.monotonic()-warm<5:
            _,a,b=group();times.append(b-a)
        count=max(2,math.ceil(35*duty/statistics.mean(times)))
        if count>100000:raise RuntimeError('Unreasonable measurement count')
        for rep in (1,2,3):
            sampler.set_phase('warmup'+str(rep));warm=time.monotonic()
            while time.monotonic()-warm<5:group()
            sampler.set_phase('measured'+str(rep));armed['start']=None
            for i in range(count):
                out,a,b=group();wait=0;wa=wb=b
                if i+1<count and duty<1:
                    # Correct the finite schedule: no idle interval AFTER final output.
                    wait=(b-a)*(1/duty-1)*count/(count-1);wa,wb=idle(wait)
                events.append(dict(repeat=rep,group=i,forwards=8,start=a,compute_end=b,
                                   requested_wait_s=wait,wait_start=wa,wait_end=wb,terminal=i+1==count))
            end=b;start=armed['start'];sampler.set_phase('outside_window')
            if start is None or end-start<30:raise RuntimeError('Complete first-TF-to-final-output window shorter than30s')
            samples=[s for s in sampler.samples if start<=s.host_monotonic_s<end]
            info=base.telemetry.telemetry_report(samples,prefix='')
            videos=count*8*args.batch;elapsed=end-start;board=info['mean_board_power_w']
            result=dict(repeat=rep,first_tf_start=start,final_gpu_output=end,elapsed_s=elapsed,
                        groups=count,forward_calls=count*8,video_evaluations=videos,unique_videos=args.batch,
                        time16_ms=elapsed*16000/videos,energy16_j=(338.2+board)*elapsed*16/videos,
                        measurement=info,output=out.detach().cpu().tolist())
            repeats.append(result);base.old.write(folder/f'repeat{rep:02d}.json',result)
            print(json.dumps(dict(model=args.model,precision=args.precision,method=args.method,cap=args.cap,repeat=rep,util=info['mean_gpu_utilization_percent'],time16_ms=result['time16_ms'])),flush=True)
        total=sum(r['video_evaluations'] for r in repeats);elapsed=sum(r['elapsed_s'] for r in repeats)
        busy=sum(r['measurement']['samples']*r['measurement']['mean_gpu_utilization_percent'] for r in repeats)/sum(r['measurement']['samples'] for r in repeats)
        energy=sum((338.2+r['measurement']['mean_board_power_w'])*r['elapsed_s'] for r in repeats)
        report=dict(status='complete',model=args.model,precision=args.precision,method=args.method,nominal_resource_cap=args.cap if args.method=='mps' else None,
            batch=args.batch,duty=duty,repeats=repeats,time16_ms=elapsed*16000/total,energy16_j=energy*16/total,
            mean_gpu_busy_percent=busy,target50_reached=abs(busy-50)<=5 if args.method=='temporal' and duty<1 else None,
            first_tf_to_final_gpu_output=True,internal_wait_included=True,terminal_wait_excluded=True,
            cost_and_utilization_same_window=True,throughput_normalized_not_single_batch_latency=True,
            unique_videos=args.batch,accuracy_not_evaluated_in_repeated_cost_window=True,
            host_proxy_w=338.2,no_window_selection=True)
        base.old.write(folder/'report.json',report);return report
    finally:
        hook.remove();samples=sampler.stop(raise_on_error=False);base.telemetry.save_samples(folder/'telemetry.csv',samples)
        clocks.stop(folder/'clock_telemetry.csv');base.old.write(folder/'groups.json',events);sampler.check()


def worker(args):
    import torch
    if args.output.exists():raise RuntimeError('Preserve prior point')
    args.output.mkdir(parents=True);args.power_limit_w=400
    guard=MPSGuard(args) if args.method=='mps' else base.Guard(args)
    identity=dict(model=args.model,precision=args.precision,method=args.method,batch=args.batch,cap=args.cap,
                  gpu_uuid=GPU,pid=os.getpid(),sources=sources(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
    try:
        guard.check();torch.set_num_threads(8)
        if args.precision=='fp32':
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
        if args.method=='mps' and os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE')!=str(args.cap):raise RuntimeError('MPS cap must precede CUDA initialization')
        model,processor,prompt,first,transfer,compute=base.runtime(args,guard,identity)
        sms=torch.cuda.get_device_properties(0).multi_processor_count;identity['visible_multiprocessors']=sms
        if args.method=='mps' and args.cap==50:
            full=json.loads((ROOT/f'mps_{args.model}_{args.precision}_100/identity.json').read_text())['visible_multiprocessors']
            identity['visible_sm_ratio_to100']=sms/full
            if not .4<=sms/full<=.55:raise RuntimeError('MPS resource restriction not reflected in CUDA SM attribute')
        rows=[r for r in base.old.rows() if r['split']=='test'];selected=rows[:args.batch]
        cpu,positions=base.prior.prepare_batch(args.model,selected,processor,prompt);inputs=transfer(cpu);del cpu
        identity.update(sample_ids=[r['sample_id'] for r in selected],frame_positions=positions)
        base.old.write(args.output/'identity.json',identity)
        conditions=[('mps_cap'+str(args.cap),1)] if args.method=='mps' else [('unrestricted',1)]
        if args.method=='temporal':
            prior=base.ROOT/f'{args.model}_{args.precision}_util/paced/report.json';report=json.loads(prior.read_text())
            if report['model']!=args.model or report['precision']!=args.precision or report['batch']!=args.batch:raise RuntimeError('Bound duty source differs')
            identity['duty_source']=dict(path=str(prior),sha256=base.old.sha(prior));conditions.append(('busy50',report['frozen_duty']))
        for label,duty in conditions:
            complete_window(args,compute,inputs,first,args.output/label,guard,duty)
        if base.old.verify_assets(Path('/data/models')/args.model)!=identity['assets']:raise RuntimeError('Frozen assets changed')
        if sources()!=identity['sources']:raise RuntimeError('Bound execution source changed')
        guard.check();base.old.write(args.output/'status.json',dict(status='complete',time=time.time()))
    except BaseException as error:
        base.old.write(args.output/'status.json',dict(status='failed',error=repr(error),time=time.time()));raise
    finally:base.old.write(args.output/'identity.json',identity);guard.close(args.output/'process_audit.json')


def daemon(start):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,CUDA_MPS_PIPE_DIRECTORY=PIPE,CUDA_MPS_LOG_DIRECTORY=str(ROOT/'mps_logs'))
    if start:
        Path(PIPE).mkdir(mode=0o700,exist_ok=True)
        (ROOT/'mps_logs').mkdir(exist_ok=True)
        result=subprocess.run(['nvidia-cuda-mps-control','-d'],env=env,text=True,capture_output=True,timeout=10)
        if result.returncode:raise RuntimeError('Own private MPS startup failed: '+result.stderr)
    else:
        for server in numeric_lines(mps('get_server_list')):
            if numeric_lines(mps('get_client_list '+str(server))):raise RuntimeError('Refuse daemon shutdown with active client')
        mps('quit')
        for _ in range(15):
            if not base.guard_records(GPU):return
            time.sleep(1)
        raise RuntimeError('GPU not released after own MPS shutdown')


def suite(args):
    if ROOT.exists():raise RuntimeError('New immutable suite only')
    ROOT.mkdir();state=dict(status='running',deadline=args.deadline,completed=[],unavailable=[dict(model='deepseek',precision='fp32',reason='109920536992 FP32 weight bytes exceed80GB; no offload')]);active=False
    def save():base.old.write(ROOT/'suite_status.json',state)
    # Complete native80GB work first, then the feasible FP32 model. No laboratory wait.
    plan=[('mps','deepseek','native',100),('mps','deepseek','native',50),('mps','kimi','native',100),('mps','kimi','native',50),
          ('temporal','deepseek','native',100),('temporal','kimi','native',100),
          ('mps','kimi','fp32',100),('mps','kimi','fp32',50),('temporal','kimi','fp32',100)]
    try:
        for method,model,precision,cap in plan:
            if args.deadline-time.time()<180:raise TimeoutError('Preserve transfer time; no new point')
            if active and method!='mps':daemon(False);active=False
            if not active and method=='mps':
                if base.guard_records(GPU):raise RuntimeError('Foreign GPU process')
                daemon(True);active=True
            batch=1 if method=='mps' or precision=='fp32' else {'deepseek':2,'kimi':6}[model]
            name=f'{method}_{model}_{precision}_{cap}';output=ROOT/name
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME='/data/cache',TMPDIR='/data/tmp',OMP_NUM_THREADS='8')
            if method=='mps':env.update(CUDA_MPS_PIPE_DIRECTORY=PIPE,CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=str(cap))
            else:env.pop('CUDA_MPS_PIPE_DIRECTORY',None);env.pop('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE',None)
            cmd=[f'/data/venvs/{model}/bin/python','-u','-m',__package__+'.rental_busy_resource_20261020',
                 '--method',method,'--model',model,'--precision',precision,'--cap',str(cap),'--batch',str(batch),
                 '--deadline',str(args.deadline),'--output',str(output)]
            base.old.write(ROOT/(name+'.command.json'),cmd)
            with (ROOT/(name+'.log')).open('x') as log:
                child=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state.update(phase=name,child_pid=child.pid,mps_active=active);save()
                try:code=child.wait(timeout=max(1,args.deadline-time.time()))
                except BaseException:
                    os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=15);raise
            if code:raise RuntimeError(f'{name} failed exit{code}; no blind retry')
            state['completed'].append(name);save()
        state.update(status='complete',finished=time.time())
    except BaseException as error:state.update(status='stopped',error=repr(error));raise
    finally:
        if active:
            try:daemon(False);state['mps_active']=False
            except BaseException as error:state['mps_shutdown_error']=repr(error)
        save()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suite',action='store_true')
    p.add_argument('--method',choices=('temporal','mps'));p.add_argument('--model',choices=('deepseek','kimi'))
    p.add_argument('--precision',choices=('native','fp32'));p.add_argument('--cap',type=int,choices=(50,100),default=100)
    p.add_argument('--batch',type=int);p.add_argument('--deadline',type=float,required=True);p.add_argument('--output',type=Path,default=ROOT)
    args=p.parse_args();args.gpu_uuid=GPU;args.output=args.output.resolve()
    if not REPO.is_relative_to('/data') or not args.output.is_relative_to(ROOT) or not 0<args.deadline-time.time()<=7200:raise ValueError('Canonical path and original finite rental only')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU:raise ValueError('Exact physical80GB GPU required')
    (suite if args.suite else worker)(args)


if __name__=='__main__':main()
