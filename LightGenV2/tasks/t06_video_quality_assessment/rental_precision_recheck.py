"""Bounded frozen rental precision comparison; independent busy and pure cost.

Existing native runs are immutable. No training, CPU offload, Graph, quantization,
power mutation, test-selected checkpoint, or repeated formal-window selection.
"""
from __future__ import annotations
import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import threading
import time
from . import rental_vlm_temporal as old
from . import rental_vlm_batch_followup as prior
from . import batch_telemetry as telemetry

TASK=Path(__file__).resolve().parent
REPO=TASK.parents[2]
ROOT=TASK/'runs/simulation/lgvq_rental_precision_20261019'
CONFIG=TASK/'configs/baselines/rental_precision_20261019.json'


def config():
    c=json.loads(CONFIG.read_text())
    assert c['models']=={'deepseek':2,'kimi':6}
    assert (c['test_videos'],c['cost_videos'],c['queue_depth'],c['repeats'])==(558,200,8,3)
    return c


def sources():
    return {**old.sources(),**prior.sources(),str(Path(__file__).relative_to(REPO)):old.sha(Path(__file__)),
            str(CONFIG.relative_to(REPO)):old.sha(CONFIG)}


def plan(batch):
    if not 1<=batch<=10:raise ValueError('Bounded original video batch')
    return [(i,min(i+batch,end)) for start,end in ((0,200),(200,558)) for i in range(start,end,batch)]


def idle_seconds(compute_s,duty):
    if compute_s<=0 or not .05<=duty<=1:raise ValueError('Positive computation and valid duty')
    return compute_s*(1/duty-1)


def fp32_weight_bytes(parameters):return parameters*4


def guard_records(gpu):
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
    return [r for r in csv.reader(raw.splitlines()) if len(r)==3 and r[1].strip()==gpu]


class Guard:
    def __init__(self,args):
        self.args=args;self.expected=None;self.audit=[];self.error=None
        self.stop=threading.Event();self.bound=sources();self.thread=None
    def check(self):
        if time.time()>=self.args.deadline:raise TimeoutError('Finite rental deadline')
        if sources()!=self.bound:raise RuntimeError('Bound source changed')
        rows=guard_records(self.args.gpu_uuid)
        self.audit.append(dict(unix=time.time(),processes=rows))
        if {r[0].strip() for r in rows}!=(self.expected or set()):raise RuntimeError('Foreign GPU process')
    def bind(self):
        rows=guard_records(self.args.gpu_uuid)
        if len(rows)!=1:raise RuntimeError('Cannot map sole GPU PID')
        self.expected={rows[0][0].strip()};self.check()
        def monitor():
            while not self.stop.wait(1):
                try:self.check()
                except BaseException as error:self.error=error;return
        self.thread=threading.Thread(target=monitor,daemon=True);self.thread.start()
    def fast(self):
        if self.error:raise RuntimeError('Independent guard failed') from self.error
        if time.time()>=self.args.deadline:raise TimeoutError('Finite rental deadline')
    def close(self,path):
        self.stop.set()
        if self.thread:self.thread.join(5)
        old.write(path,self.audit)


def runtime(args,guard,identity):
    import torch
    train=old.RUN/args.model/'train'
    receipt=json.loads((train/'training_receipt.json').read_text())
    ptpath=train/'best_checkpoint.pt'
    if old.sha(ptpath)!=receipt['artifacts']['best_checkpoint.pt']:raise RuntimeError('Frozen head SHA')
    assets=old.verify_assets(Path('/data/models')/args.model)
    if assets!=receipt['identity']['model_sha256']:raise RuntimeError('Original official weights')
    # Training source is a historical identity, not the latest unrelated main HEAD.
    original=receipt['identity']
    for path,digest in original['source_sha256'].items():
        blob=subprocess.check_output(['git','show',original['git_commit']+':'+path.replace('\\','/')],cwd=REPO)
        if hashlib.sha256(blob).hexdigest()!=digest:raise RuntimeError('Historical training-source SHA')
    parameters=original['runtime']['total_model_parameters']
    identity.update(original_training_commit=original['git_commit'],assets=assets,
        checkpoint_sha256=old.sha(ptpath),training_receipt_sha256=old.sha(train/'training_receipt.json'),
        original_parameters=parameters,fp32_weight_bytes=fp32_weight_bytes(parameters))
    free,total=torch.cuda.mem_get_info()
    if args.precision=='fp32' and fp32_weight_bytes(parameters)>free:
        raise MemoryError('FP32 weights alone exceed free VRAM; no CUDA model allocation or offload')
    processor,prompt,ids=old.processor(args.model,Path('/data/models')/args.model)
    if args.precision=='native':
        model,language,first=old.load_model(args.model,Path('/data/models')/args.model)
        actual=old.runtime_identity(model)
        if actual!=original['runtime']:raise RuntimeError('Native original runtime changed')
    else:
        from transformers import AutoConfig,AutoModelForCausalLM
        cfg=AutoConfig.from_pretrained('/data/models/'+args.model,trust_remote_code=True,local_files_only=True)
        if args.model=='kimi':
            cfg.vision_config._attn_implementation='eager';cfg.text_config._attn_implementation='eager'
        model=AutoModelForCausalLM.from_pretrained('/data/models/'+args.model,config=cfg,
            local_files_only=True,trust_remote_code=True,torch_dtype=torch.float32,
            attn_implementation='eager',low_cpu_mem_usage=True).eval().requires_grad_(False).cuda()
        language=model.language if args.model=='deepseek' else model.language_model
        first=model.vision.blocks[0] if args.model=='deepseek' else model.vision_tower.encoder.blocks[0]
        actual=old.runtime_identity(model)
        if any(p.is_floating_point() and p.dtype!=torch.float32 for p in model.parameters()):
            raise RuntimeError('Not all floating model parameters are FP32')
    guard.bind()
    pt=torch.load(ptpath,map_location='cpu',weights_only=False)
    if pt['quality_token_ids']!=ids:raise RuntimeError('Original five quality words')
    head=old.five_rows(pt['state_dict']['weight']).cuda().eval().requires_grad_(False)
    levels=pt['level_scores'].cuda().float()
    dtype=torch.bfloat16 if args.precision=='native' else torch.float32
    def transfer(cpu):return {k:v.to('cuda',dtype=dtype if v.is_floating_point() else v.dtype) for k,v in cpu.items()}
    def compute(inputs):
        with torch.inference_mode():
            if args.precision=='native':
                value=prior.hidden(args.model,model,language,inputs)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    native=language.lm_head(value.to(language.lm_head.weight.dtype)).float()
            else:
                if args.model=='deepseek':embeds=model.prepare_inputs_embeds(**inputs)
                else:
                    embeds=model.get_input_embeddings()(inputs['input_ids'])
                    visual=model._extract_image_features(inputs['pixel_values'],inputs['image_grid_hws'])
                    embeds=model._merge_with_image_features(embeds,inputs['input_ids'],visual)
                value=language.model(inputs_embeds=embeds,attention_mask=inputs['attention_mask'],use_cache=False,return_dict=True).last_hidden_state[:,-1].float()
                native=language.lm_head(value).float()
            mos=head(value).softmax(-1)@levels
        return mos,native
    identity.update(runtime=actual,precision=args.precision,
        fp32_caveat='Existing deployment weights cast to FP32, not recovery of quantized values' if args.precision=='fp32' else None,
        fp32_attention='official eager, since FlashAttention2 does not support FP32' if args.precision=='fp32' else None,
        model_floating_dtypes=sorted({str(p.dtype) for p in model.parameters() if p.is_floating_point()}))
    return model,processor,prompt,first,transfer,compute


def measure_util(args,compute,inputs,folder,guard,paced):
    import torch
    folder.mkdir();sampler=telemetry.TelemetrySampler(args.gpu_uuid,raw_path=folder/'telemetry_stdout.jsonl')
    clocks=prior.ClockSampler(args.gpu_uuid);groups=[];updates=[];repeats=[]
    sampler.start();clocks.start()
    def wait(seconds):
        until=time.monotonic()+seconds
        while time.monotonic()<until:
            guard.fast();sampler.check();time.sleep(min(.1,max(0,until-time.monotonic())))
    def phase(name,minimum,duty,adaptive=False):
        start=time.monotonic();last=start;sampler.set_phase(name)
        while time.monotonic()-start<minimum:
            guard.fast();sampler.check();tick=time.monotonic()
            for _ in range(8):out,_=compute(inputs)
            torch.cuda.synchronize();computed=time.monotonic()
            delay=idle_seconds(computed-tick,duty) if paced else 0
            wait(delay);end=time.monotonic()
            groups.append(dict(phase=name,start_monotonic_s=tick,compute_end_monotonic_s=computed,
                               end_monotonic_s=end,requested_idle_s=delay,duty=duty,forwards=8))
            if adaptive and end-last>=5:
                values=[s.utilization_percent for s in sampler.samples if last<=s.host_monotonic_s<end]
                if len(values)>=10:
                    observed=statistics.mean(values);new=min(1,max(.05,duty*min(1.25,max(.8,50/max(observed,1)))))
                    updates.append(dict(start=last,end=end,samples=len(values),observed=observed,old=duty,new=new));duty=new;last=end
        return start,time.monotonic(),out,duty
    try:
        duty=.5 if paced else 1
        if paced:_,_,_,duty=phase('calibration',60,duty,True)
        for i in range(1,4):
            ws,we,_,_=phase('warmup'+str(i),5,duty)
            start,end,out,_=phase('measured'+str(i),30,duty)
            selected=[s for s in sampler.samples if start<=s.host_monotonic_s<end]
            summary=telemetry.telemetry_report(selected,prefix='')
            record=dict(repeat=i,start_monotonic_s=start,end_monotonic_s=end,duration_s=end-start,
                warmup_s=we-ws,frozen_duty=duty,measurement=summary,output=out.detach().cpu().tolist())
            repeats.append(record);old.write(folder/f'repeat{i:02d}.json',record)
        pooled=sum(r['measurement']['samples']*r['measurement']['mean_gpu_utilization_percent'] for r in repeats)/sum(r['measurement']['samples'] for r in repeats)
        report=dict(status='complete',model=args.model,batch=args.batch,precision=args.precision,paced=paced,
            repeats=repeats,mean_gpu_busy_percent=pooled,frozen_duty=duty,target50_reached=abs(pooled-50)<=5 if paced else None,
            no_window_selection=True,independent_of_dataset_once_cost=True)
        old.write(folder/'report.json',report);return report
    finally:
        samples=sampler.stop(raise_on_error=False);telemetry.save_samples(folder/'telemetry.csv',samples)
        clocks.stop(folder/'clock_telemetry.csv');old.write(folder/'forward_groups.json',groups)
        old.write(folder/'controller_updates.json',updates);sampler.check()


def evaluate(args,compute,prepare,transfer,first,folder,guard,duty,identity):
    import numpy as np
    import torch
    from . import quality_token_common as core
    from LightGenV2.common.baseline_measurement import FirstBlockTimer
    from LightGenV2.common import baseline_measurement as bm
    test=[r for r in old.rows() if r['split']=='test'];folder.mkdir()
    sampler=telemetry.TelemetrySampler(args.gpu_uuid,raw_path=folder/'telemetry_stdout.jsonl')
    clocks=prior.ClockSampler(args.gpu_uuid);timer=FirstBlockTimer(first)
    timings=[];predictions=[];waits=[]
    sampler.start();clocks.start();sampler.set_phase('idle');time.sleep(2);sampler.set_phase(None)
    try:
        with (folder/'predictions.csv').open('x',newline='') as ps,(folder/'batch_timing.csv').open('x',newline='') as ts:
            pw,tw=csv.writer(ps),csv.writer(ts)
            pw.writerow(['sample_index','sample_id','target_mos','prediction','frame_positions'])
            tw.writerow(['batch_index','videos','in_cost_subset','synchronized_wall_ms','cuda_ms','preprocessing_audit_ms','h2d_audit_ms'])
            for index,(start,end) in enumerate(plan(args.batch)):
                guard.fast();sampler.check();tick=time.perf_counter();cpu,positions=prepare(test[start:end]);prep=(time.perf_counter()-tick)*1000
                tick=time.perf_counter();inputs=transfer(cpu);torch.cuda.synchronize();h2d=(time.perf_counter()-tick)*1000
                timer.reset();sampler.set_phase(f'active:{index}' if end<=200 else None)
                try:mos,native=compute(inputs);measured=timer.finish()
                finally:sampler.set_phase(None)
                values=mos.detach().cpu().tolist()
                if not all(math.isfinite(v) for v in values):raise RuntimeError('Nonfinite MOS')
                predictions.extend(values);record=[index,end-start,end<=200,measured['host_ms'],measured['cuda_ms'],prep,h2d]
                timings.append(record);tw.writerow(record)
                for j,value in enumerate(values):pw.writerow([start+j,test[start+j]['sample_id'],test[start+j]['temporal'],value,json.dumps(positions[j])])
                ps.flush();ts.flush()
                requested=idle_seconds(measured['host_ms']/1000,duty);tick=time.monotonic();until=tick+requested
                sampler.set_phase('excluded_wait')
                while time.monotonic()<until:
                    guard.fast();sampler.check();time.sleep(min(.1,max(0,until-time.monotonic())))
                waits.append(dict(batch_index=index,in_cost_subset=end<=200,requested_s=requested,start=tick,end=time.monotonic()))
                sampler.set_phase(None);del cpu,inputs,mos,native
                old.write(folder/'status.json',dict(status='evaluating',videos=end,time=time.time()))
                if index%20==0 or end==558:print(json.dumps(dict(stage=args.stage,model=args.model,precision=args.precision,videos=end)),flush=True)
        samples=sampler.stop();telemetry.save_samples(folder/'telemetry.csv',samples)
        cost=[r for r in timings if r[2]];power=bm.power_report(samples,[r[3] for r in cost],power_limit_w=400)
        report=dict(status='complete',identity=identity,test_videos=558,cost_videos=200,
            performance=core.metrics(np.array([r['temporal'] for r in test]),np.array(predictions)),
            cost=telemetry.workload_cost(cost,power['active_mean_w']),power=power,
            explicit_warmup_forwards=0,first_batch_included=True,frozen_duty=duty,
            boundary='first native vision block input to synchronized GPU MOS output',
            excluded=['decode','processor','H2D','loading','inter-inference imposed waits'],
            independent_utilization_source=args.utilization_source)
        old.write(folder/'report.json',report);old.write(folder/'status.json',dict(status='complete',videos=558))
    finally:
        if sampler.process is not None:telemetry.save_samples(folder/'telemetry.csv',sampler.stop(raise_on_error=False))
        clocks.stop(folder/'clock_telemetry.csv');timer.close();old.write(folder/'excluded_waits.json',waits);sampler.check()


def worker(args):
    import torch
    if args.output.exists():raise RuntimeError('Preserve existing evidence')
    args.output.mkdir(parents=True);guard=Guard(args)
    identity=dict(model=args.model,batch=args.batch,precision=args.precision,stage=args.stage,
        sources=sources(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        gpu_uuid=args.gpu_uuid,pid=os.getpid(),deadline=args.deadline,protocol=config())
    try:
        guard.check();torch.set_num_threads(8)
        gpu=subprocess.check_output(['nvidia-smi',f'--id={args.gpu_uuid}',
            '--query-gpu=name,memory.free,power.limit','--format=csv,noheader,nounits'],text=True)
        name,free,watts=next(csv.reader(gpu.splitlines()))
        if 'A100-SXM4-80GB' not in name or float(watts)!=400 or float(free)<75000:
            raise RuntimeError('Original free80GB/unchanged400W admission required')
        identity['gpu']=gpu.strip()
        if args.precision=='fp32':
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.set_float32_matmul_precision('highest')
        model,p,prompt,first,transfer,compute=runtime(args,guard,identity)
        old.write(args.output/'identity.json',identity)
        prepare=lambda rows:prior.prepare_batch(args.model,rows,p,prompt)
        if args.stage=='util':
            test=[r for r in old.rows() if r['split']=='test']
            cpu,positions=prepare(test[:args.batch]);inputs=transfer(cpu)
            old.write(args.output/'input_identity.json',dict(sample_ids=[r['sample_id'] for r in test[:args.batch]],positions=positions,
                tensors={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()) for k,v in cpu.items() if torch.is_tensor(v)}))
            reference,_=compute(inputs);reference=reference.detach().cpu()
            for paced in (False,True):
                report=measure_util(args,compute,inputs,args.output/('paced' if paced else 'native'),guard,paced)
                if any(not torch.allclose(torch.tensor(r['output']),reference,atol=.0002,rtol=0) for r in report['repeats']):raise RuntimeError('Repeated MOS changed')
                print(json.dumps(dict(model=args.model,precision=args.precision,paced=paced,util=report['mean_gpu_busy_percent'])),flush=True)
        else:
            util=json.loads(Path(args.utilization_source).read_text())
            if util['status']!='complete' or util['model']!=args.model or util['precision']!=args.precision or util['batch']!=args.batch:raise RuntimeError('Utilization identity')
            duty=util['frozen_duty'] if args.stage=='paced-cost' else 1
            identity['independent_utilization']=dict(path=args.utilization_source,sha256=old.sha(Path(args.utilization_source)),value=util['mean_gpu_busy_percent'])
            evaluate(args,compute,prepare,transfer,first,args.output/'evaluation',guard,duty,identity)
        if old.verify_assets(Path('/data/models')/args.model)!=identity['assets']:raise RuntimeError('Assets changed')
        guard.check();old.write(args.output/'status.json',dict(status='complete',time=time.time()))
    except (MemoryError,torch.cuda.OutOfMemoryError) as error:
        old.write(args.output/'status.json',dict(status='capacity_unavailable',error=str(error),identity=identity,no_retry=True));raise
    except BaseException as error:
        old.write(args.output/'status.json',dict(status='failed',error=repr(error),time=time.time()));raise
    finally:
        old.write(args.output/'identity.json',identity);guard.close(args.output/'process_audit.json')


def suite(args):
    import fcntl
    if ROOT.exists():raise RuntimeError('Preserve existing suite evidence')
    with (TASK/'runs/simulation/.rental_precision.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ROOT.mkdir();bound=sources()
        state=dict(status='running',started=time.time(),deadline=args.deadline,completed=[],unavailable=[],sources=bound)
        def save():old.write(ROOT/'suite_status.json',state)
        try:
            for precision,model,batch in [('native','deepseek',2),('native','kimi',6),('fp32','kimi',1),('fp32','deepseek',1)]:
                for stage in ('util','paced-cost','native-cost'):
                    if sources()!=bound:raise RuntimeError('Suite source changed')
                    if args.deadline-time.time()<240:raise TimeoutError('Preserve final backup reserve')
                    for _ in range(3):
                        if guard_records(args.gpu_uuid):raise RuntimeError('Foreign GPU at admission')
                        time.sleep(3)
                    name=f'{model}_{precision}_{stage}'
                    target=ROOT/name;utilstage='paced' if stage=='paced-cost' else 'native'
                    util=ROOT/f'{model}_{precision}_util'/utilstage/'report.json'
                    command=[f'/data/venvs/{model}/bin/python','-u','-m',__package__+'.rental_precision_recheck',
                        '--model',model,'--precision',precision,'--stage',stage,'--batch',str(batch),
                        '--gpu-uuid',args.gpu_uuid,'--deadline',str(args.deadline),'--output',str(target)]
                    if stage!='util':command+=['--utilization-source',str(util)]
                    old.write(ROOT/(name+'.command.json'),command)
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu_uuid,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME='/data/cache',TMPDIR='/data/tmp',OMP_NUM_THREADS='8')
                    with (ROOT/(name+'.log')).open('x') as log:
                        child=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        state.update(phase=name,child_pid=child.pid);save()
                        try:code=child.wait(timeout=max(1,args.deadline-time.time()))
                        except BaseException:
                            os.killpg(child.pid,signal.SIGTERM)
                            try:child.wait(timeout=10)
                            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
                            raise
                    terminal=json.loads((target/'status.json').read_text()) if (target/'status.json').exists() else {}
                    if code:
                        if terminal.get('status')=='capacity_unavailable':
                            state['unavailable'].append(dict(stage=name,evidence=str(target/'status.json')));save();break
                        raise RuntimeError(f'{name} failed exit{code}, no automatic retry')
                    state['completed'].append(name);save()
            state.update(status='complete',finished=time.time())
        except BaseException as error:state.update(status='failed',error=repr(error));raise
        finally:save()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',action='store_true')
    p.add_argument('--model',choices=('deepseek','kimi'))
    p.add_argument('--precision',choices=('native','fp32'))
    p.add_argument('--stage',choices=('util','native-cost','paced-cost'))
    p.add_argument('--batch',type=int);p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--deadline',type=float,required=True);p.add_argument('--output',type=Path,default=ROOT)
    p.add_argument('--utilization-source')
    args=p.parse_args();args.output=args.output.resolve()
    if not args.output.is_relative_to(ROOT) or not REPO.is_relative_to('/data') or not 0<args.deadline-time.time()<=7200:raise ValueError('Canonical rental run and finite2h required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=args.gpu_uuid:raise ValueError('Exact physical GPU')
    if args.suite:suite(args)
    else:worker(args)


if __name__=='__main__':main()
