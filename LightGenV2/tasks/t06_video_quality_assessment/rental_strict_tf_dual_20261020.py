"""Frozen Kimi/full-DS strict timing; explicit single/dual-GPU resource profiles."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from . import rental_precision_recheck as base
from . import rental_busy_resource_20261020 as old_busy
from . import strict_tf_window_20261020 as timing

TASK=base.TASK;REPO=base.REPO
CONFIG=TASK/'configs/baselines/rental_strict_tf_dual_20261020.json'
ROOT=TASK/'runs/simulation/lgvq_strict_tf_rental_dual_20261020'
PIPE='/data/tmp/t06_strict_tf_dual_20261020'


def config():return json.loads(CONFIG.read_text())


def sources():
    return {**base.sources(),**old_busy.sources(),**{str(p.relative_to(REPO)):base.old.sha(p)
        for p in (Path(__file__),Path(timing.__file__),CONFIG)}}


def placement(model,precision):
    return config()['gpu_uuids'][:2 if model=='deepseek' and precision=='fp32' else 1]


def mps(command):
    result=subprocess.run(['nvidia-cuda-mps-control'],input=command+'\n',text=True,capture_output=True,
        timeout=8,env=dict(os.environ,CUDA_MPS_PIPE_DIRECTORY=PIPE))
    if result.returncode:raise RuntimeError('Private MPS control failed: '+result.stdout+result.stderr)
    return result.stdout


def records():
    allowed=set(config()['gpu_uuids'])
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
    return [list(map(str.strip,row)) for row in csv.reader(raw.splitlines()) if len(row)==3 and row[1].strip() in allowed]


class Guard:
    def __init__(self,args):
        self.args=args;self.bound=sources();self.audit=[];self.expected=None;self.error=None
        self.stop=threading.Event();self.thread=None
        line=next((s for s in Path('/proc/self/status').read_text().splitlines() if s.startswith('NSpid:')),'')
        self.ids={os.getpid(),*[int(x) for x in line.split()[1:]]}
    def check(self):
        if time.time()>=self.args.deadline:raise TimeoutError('Authorized fixed compute deadline')
        if sources()!=self.bound:raise RuntimeError('Published bound source changed')
        rows=records();servers=set();clients=set()
        if self.args.method=='mps':
            servers=old_busy.numeric_lines(mps('get_server_list'))
            for server in servers:clients|=old_busy.numeric_lines(mps('get_client_list '+str(server)))
            if clients-self.ids:raise RuntimeError('Foreign private-MPS client')
            if self.expected is not None and not clients&self.ids:raise RuntimeError('Own MPS client missing')
            if {int(r[0]) for r in rows}-(servers|self.ids):raise RuntimeError('Foreign non-MPS GPU process')
        elif self.expected is None:
            if rows:raise RuntimeError('Allocated GPUs must be idle before CUDA')
        elif {r[0] for r in rows}!=self.expected:raise RuntimeError('Foreign process or own context missing')
        if self.expected is not None and {r[1] for r in rows}!=set(self.args.gpu_uuids):
            raise RuntimeError('GPU process-to-device placement differs')
        self.audit.append(dict(unix=time.time(),processes=rows,servers=sorted(servers),clients=sorted(clients)))
    def bind(self):
        rows=records()
        if self.args.method!='mps' and len({r[0] for r in rows})!=1:raise RuntimeError('Expected one dispatched-model process')
        self.expected={r[0] for r in rows};self.check()
        def monitor():
            while not self.stop.wait(1):
                try:self.check()
                except BaseException as error:self.error=error;return
        self.thread=threading.Thread(target=monitor,daemon=True);self.thread.start()
    def fast(self):
        if self.error:raise RuntimeError('Independent GPU/source guard failed') from self.error
        if time.time()>=self.args.deadline:raise TimeoutError('Authorized fixed deadline')
    def close(self,path):
        self.stop.set()
        if self.thread:self.thread.join(8)
        base.old.write(path,self.audit)


def smoke(args,compute,inputs,first,folder,guard):
    """GPU boundary audit with a known PRE-TF delay, not performance data."""
    import torch
    def sync():
        for i in range(len(args.gpu_uuids)):torch.cuda.synchronize(i)
    state={};result=[]
    def start(*_):
        if state.get('start') is None:sync();state['start']=time.monotonic()
    hook=first.register_forward_pre_hook(start)
    try:
        for delay in (0.,.05,0.):
            guard.fast();sync();state['start']=None;before=time.monotonic();time.sleep(delay)
            out,native=compute(inputs);sync();end=time.monotonic();tf=state['start']
            if tf is None or tf-before<delay-.001:raise RuntimeError('First-block boundary does not exclude known frontend delay')
            if not bool(torch.isfinite(out).all()) or not bool(torch.isfinite(native).all()):raise RuntimeError('Nonfinite smoke output')
            output=out.detach().cpu().tolist()
            if result and any(abs(a-b)>2e-4 for a,b in zip(output,result[0]['output'])):raise RuntimeError('Frozen-input smoke outputs differ')
            result.append(dict(injected_preblock_s=delay,call_start=before,first_block=tf,output_ready=end,
                excluded_preblock_s=tf-before,charged_compute_s=end-tf,output=output,
                native_logits_shape=list(native.shape)))
        base.old.write(folder/'boundary_smoke.json',dict(status='passed',audit_only_not_formal=True,
            all_used_devices_synchronized=True,known_frontend_delay_excluded=True,calls=result))
    finally:hook.remove()


def accuracy(args,processor,prompt,transfer,compute,folder,guard,identity):
    import torch
    import numpy as np
    from . import quality_token_common as core
    rows=[r for r in base.old.rows() if r['split']=='test']
    if len(rows)!=558:raise RuntimeError('Original558 TEST required')
    folder.mkdir();predictions=[];targets=[];start=time.monotonic()
    with (folder/'predictions.csv').open('x',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['sample_index','sample_id','target_mos','prediction','frame_positions'])
        for i,row in enumerate(rows):
            guard.fast();cpu,positions=base.prior.prepare_batch(args.model,[row],processor,prompt);inputs=transfer(cpu)
            out,_=compute(inputs)
            for device in range(len(args.gpu_uuids)):torch.cuda.synchronize(device)
            prediction=float(out.detach().cpu()[0]);target=float(row['temporal'])
            if not __import__('math').isfinite(prediction):raise RuntimeError('Nonfinite accuracy prediction')
            predictions.append(prediction);targets.append(target)
            writer.writerow([i,row['sample_id'],target,prediction,json.dumps(positions)]);stream.flush()
            if (i+1)%25==0:
                base.old.write(folder/'progress.json',dict(samples=i+1,total=558,elapsed_s=time.monotonic()-start))
                print(f'full558 {i+1}/558',flush=True)
    report=dict(status='complete',samples=558,model=args.model,precision=args.precision,batch=1,
        metrics=core.metrics(np.array(targets),np.array(predictions)),checkpoint_sha256=identity['checkpoint_sha256'],
        assets=identity['assets'],accuracy_independent_of_cost=True,
        selection='original TEST-selected frozen head; development estimate, noVAL')
    base.old.write(folder/'report.json',report);return report


def worker(args):
    import torch
    if args.output.exists():raise RuntimeError('Preserve all existing point evidence')
    args.output.mkdir();guard=Guard(args);identity=dict(model=args.model,precision=args.precision,
        method=args.method,cap=args.cap,batch=args.batch,gpu_uuids=args.gpu_uuids,pid=os.getpid(),
        sources=sources(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        timing_protocol=timing.PROTOCOL,config_sha256=base.old.sha(CONFIG))
    try:
        guard.check();torch.set_num_threads(8)
        for gpu in args.gpu_uuids:
            power=float(subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-gpu=power.limit','--format=csv,noheader,nounits'],text=True))
            if abs(power-400)>.1:raise RuntimeError('Original400W platform required, no power mutation')
        if args.method=='mps' and os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE')!=str(args.cap):raise RuntimeError('MPS cap must precede CUDA')
        if args.precision=='fp32':
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.set_float32_matmul_precision('highest')
        model,processor,prompt,first,transfer,compute=base.runtime(args,guard,identity)
        properties=[torch.cuda.get_device_properties(i).multi_processor_count for i in range(len(args.gpu_uuids))]
        identity['visible_multiprocessors']=properties
        if args.method=='mps':
            expected=108 if args.cap==100 else 54
            if properties!=[expected]*len(args.gpu_uuids):raise RuntimeError('Actual MPS SM allocation differs')
        identity['parameter_devices']=sorted({str(p.device) for p in model.parameters()})
        identity['parameter_dtypes']=sorted({str(p.dtype) for p in model.parameters() if p.is_floating_point()})
        rows=[r for r in base.old.rows() if r['split']=='test']
        cpu,positions=base.prior.prepare_batch(args.model,rows[:args.batch],processor,prompt);inputs=transfer(cpu)
        identity.update(sample_ids=[r['sample_id'] for r in rows[:args.batch]],frame_positions=positions)
        base.old.write(args.output/'identity.json',identity)
        smoke(args,compute,inputs,first,args.output,guard)
        conditions=[('mps_cap'+str(args.cap),1)] if args.method=='mps' else [('unrestricted',1),('busy50',.5)]
        for label,duty in conditions:
            report=timing.measure(args,compute,inputs,first,args.output/label,guard,duty)
            if report['target50_reached'] is False:raise RuntimeError('Complete3 trials outside50+/-5; preserve without selecting/retrying')
        if args.accuracy:
            accuracy(args,processor,prompt,transfer,compute,args.output/'accuracy558',guard,identity)
        if base.old.verify_assets(Path('/data/models')/args.model)!=identity['assets']:raise RuntimeError('Official assets changed')
        guard.check();base.old.write(args.output/'status.json',dict(status='complete',finished=time.time()))
    except BaseException as error:
        base.old.write(args.output/'status.json',dict(status='failed',error=repr(error),time=time.time()));raise
    finally:base.old.write(args.output/'identity.json',identity);guard.close(args.output/'process_audit.json')


def daemon(gpus,start):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(gpus),CUDA_MPS_PIPE_DIRECTORY=PIPE,
        CUDA_MPS_LOG_DIRECTORY=str(ROOT/'mps_logs'))
    if start:
        Path(PIPE).mkdir(mode=0o700,exist_ok=True);(ROOT/'mps_logs').mkdir(exist_ok=True)
        result=subprocess.run(['nvidia-cuda-mps-control','-d'],env=env,capture_output=True,text=True,timeout=10)
        if result.returncode:raise RuntimeError('Private MPS startup failed: '+result.stderr)
    else:
        for server in old_busy.numeric_lines(mps('get_server_list')):
            if old_busy.numeric_lines(mps('get_client_list '+str(server))):raise RuntimeError('Refuse MPS exit with clients')
        mps('quit')
        for _ in range(15):
            if not records():return
            time.sleep(1)
        raise RuntimeError('GPU processes remain after own daemon exit')


def suite(args):
    if ROOT.exists():raise RuntimeError('Preserve existing suite; no automatic resume/retry')
    if records():raise RuntimeError('Both allocated GPUs must be empty')
    ROOT.mkdir();state=dict(status='running',deadline=args.deadline,completed=[],sources=sources(),config=config())
    def save():base.old.write(ROOT/'suite_status.json',state)
    active=False;active_gpus=None
    # DS FP32 first because it is the newly authorized, previously missing case.
    plan=[(model,precision,method,cap) for model,precision in
        (('deepseek','fp32'),('kimi','fp32'),('deepseek','native'),('kimi','native'))
        for method,cap in (('mps',100),('mps',50),('temporal',100))]
    try:
        for model,precision,method,cap in plan:
            if time.time()>=args.deadline-240:raise TimeoutError('No new point during backup reserve')
            gpus=placement(model,precision)
            if active and (method!='mps' or active_gpus!=gpus):daemon(active_gpus,False);active=False
            if method=='mps' and not active:
                if records():raise RuntimeError('Foreign GPU process before own MPS start')
                daemon(gpus,True);active=True;active_gpus=gpus
            batch=1 if method=='mps' else config()['temporal_batches'][model+'_'+precision]
            name=f'{method}_{model}_{precision}_{cap}';output=ROOT/name
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(gpus),OMP_NUM_THREADS='8',
                HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME='/data/cache',TMPDIR='/data/tmp')
            if method=='mps':env.update(CUDA_MPS_PIPE_DIRECTORY=PIPE,CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=str(cap))
            else:env.pop('CUDA_MPS_PIPE_DIRECTORY',None);env.pop('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE',None)
            cmd=[f'/data/venvs/{model}/bin/python','-u','-m',__package__+'.rental_strict_tf_dual_20261020',
                '--model',model,'--precision',precision,'--method',method,'--cap',str(cap),'--batch',str(batch),
                '--deadline',str(args.deadline),'--output',str(output),'--attempt',str(args.attempt)]
            if model=='deepseek' and precision=='fp32' and method=='mps' and cap==100:cmd.append('--accuracy')
            base.old.write(ROOT/(name+'.command.json'),cmd)
            with (ROOT/(name+'.log')).open('x') as log:
                child=subprocess.Popen(cmd,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state.update(phase=name,child_pid=child.pid);save()
                try:code=child.wait(timeout=max(1,args.deadline-time.time()))
                except BaseException:
                    os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=20);raise
            if code:raise RuntimeError(f'{name} failed exit{code}; preserve all, no automatic retry')
            state['completed'].append(name);save()
        state.update(status='complete',finished=time.time())
    except BaseException as error:state.update(status='stopped',error=repr(error));raise
    finally:
        if active:
            try:daemon(active_gpus,False);state['mps_active']=False
            except BaseException as error:state['mps_cleanup_error']=repr(error)
        save()


def main():
    global ROOT,PIPE
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suite',action='store_true')
    p.add_argument('--model',choices=('deepseek','kimi'));p.add_argument('--precision',choices=('native','fp32'))
    p.add_argument('--method',choices=('mps','temporal'));p.add_argument('--cap',type=int,choices=(50,100),default=100)
    p.add_argument('--batch',type=int);p.add_argument('--accuracy',action='store_true');p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--attempt',type=int,choices=(1,2),default=1)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.attempt==2:
        prior=json.loads((ROOT/'suite_status.json').read_text())
        if prior['status']!='stopped' or prior['completed'] or list(ROOT.glob('*/mps_cap*/report.json')):
            raise RuntimeError('Attempt2 only preserves a CUDA-loading preflight failure without formal data')
        ROOT=ROOT/'attempt02';PIPE=PIPE+'_attempt02'
    args.output=(args.output or ROOT).resolve()
    if not 0<args.deadline-time.time()<=7200 or args.deadline>config()['compute_deadline']:raise ValueError('Authorized finite rental window only')
    if not REPO.is_relative_to('/data') or not args.output.is_relative_to(ROOT):raise ValueError('Canonical rental source/run only')
    if args.suite:return suite(args)
    args.gpu_uuids=placement(args.model,args.precision);args.gpu_uuid=args.gpu_uuids[0];args.power_limit_w=400
    args.dual_fp32=len(args.gpu_uuids)==2
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=','.join(args.gpu_uuids):raise ValueError('Exact physical assigned devices required')
    expected=1 if args.method=='mps' else config()['temporal_batches'][args.model+'_'+args.precision]
    if args.batch!=expected:raise ValueError('Fixed authorized batch; no scan')
    worker(args)


if __name__=='__main__':main()
