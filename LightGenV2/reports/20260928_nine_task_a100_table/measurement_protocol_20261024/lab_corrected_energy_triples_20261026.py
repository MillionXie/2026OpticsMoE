"""New-scope serial lab baselines, three fresh rounds and matched power.

Reuse original receipt-bound runtime/first-block hooks. All image preparation
and H2D execute one video at a time but are excluded in accounting; tokenizer,
GPU frontend, native backbone/head and other actual internal waits remain.
"""
import argparse
import ast
import inspect
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace
import rental_energy_triples_20261026 as accounting

REPO=accounting.REPO
TASK=REPO/'LightGenV2/tasks/t06_video_quality_assessment'
GPU='GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55'


def diagnostic_guard_check(original):
    """Move ONLY the original audit append before validation, preserving gates.

    The old guard discarded the exact offending process snapshot on failure.
    Every original query, source/deadline/MPS/process condition stays identical.
    """
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    fn=tree.body[0]
    append=fn.body[-1]
    if not isinstance(append,ast.Expr) or not isinstance(append.value,ast.Call):raise ValueError('Guard audit AST changed')
    if ast.unparse(append.value.func)!='self.audit.append':raise ValueError('Guard audit target changed')
    prefix=fn.body[:-1]
    original_AST=ast.dump(ast.Module(body=prefix,type_ignores=[]),include_attributes=False)
    index=next(i for i,node in enumerate(prefix) if isinstance(node,ast.If) and 'clients - self.self_ids' in ast.unparse(node.test))
    fn.body=prefix[:index]+[append]+prefix[index:]
    without=[node for node in fn.body if node is not append]
    if ast.dump(ast.Module(body=without,type_ignores=[]),include_attributes=False)!=original_AST:
        raise ValueError('Original guard mathematics/queries changed')
    namespace=dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(tree),__file__+'::audit_before_guard_rejection','exec'),namespace)
    return namespace[fn.name]


def sources():
    from LightGenV2.tasks.t06_video_quality_assessment import measure_mps_e2e_20261024 as old
    paths=[Path(__file__),Path(accounting.__file__),Path(accounting.original.__file__)]
    return dict(old.source_map(),**{p.relative_to(REPO).as_posix():old.sha(p) for p in paths})


def wrap_tokenizers(prepare, records):
    """Only observe processors referenced by the native preparation closure."""
    restored=[];seen=set()
    def visit(value,depth=0):
        if id(value) in seen or depth>3:return
        seen.add(id(value))
        for name in ('tokenizer','_quality_processor'):
            item=getattr(value,name,None)
            if name=='tokenizer' and item is not None:
                setattr(value,name,accounting.original.TokenizerAudit(item,records))
                restored.append((value,name,item))
            elif item is not None:visit(item,depth+1)
        if inspect.isfunction(value):
            for item in inspect.getclosurevars(value).nonlocals.values():visit(item,depth+1)
    visit(prepare)
    return restored


def worker(args):
    import numpy as np
    import torch
    from LightGenV2.tasks.t06_video_quality_assessment import measure_mps_e2e_20261024 as old
    from LightGenV2.tasks.t06_video_quality_assessment import e2e_telemetry_observer as observer
    from LightGenV2.tasks.t06_video_quality_assessment.nvml_observer_process import ProcessTelemetrySampler
    from LightGenV2.tasks.t06_video_quality_assessment.batch_telemetry import save_samples
    if args.output.exists():raise FileExistsError('Preserve old point')
    args.output.mkdir()
    legacy=old.initialize_profile(args);bound=sources()
    snapshot_check=diagnostic_guard_check(legacy.engine.MPSGuard.check)
    class Guard(legacy.engine.MPSGuard):
        def check(self):
            if sources()!=bound:raise RuntimeError('Frozen source changed')
            try:snapshot_check(self)
            except BaseException as error:
                accounting.write(args.output/'guard_rejection_snapshot.json',dict(unix=time.time(),
                    error=repr(error),snapshot=self.audit[-1] if self.audit else None,
                    recorded_before_model_or_cleanup_unwinding=True))
                raise
    guard=Guard(args);sampler=None;finish=None;records=[];tokenizer=[];restored=[]
    identity=dict(model=args.model,cap=args.cap,repeat=args.repeat,video_calls=2048,
        unique_videos=512,batch=1,sources=bound,command=sys.argv,pid=os.getpid(),gpu_uuid=GPU,
        published_commit=os.environ['MEASUREMENT_PUBLISHED_COMMIT'],
        actual_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        no_warmup=True,no_graph=True,no_prefetch=True,CPU_preparation_single_video=True)
    def status(value,**kw):accounting.write(args.output/'status.json',dict(status=value,pid=os.getpid(),samples=len(records),unix=time.time(),**kw))
    try:
        guard.check();torch.set_num_threads(8);status('loading_original_native_model')
        if os.environ['CUDA_VISIBLE_DEVICES']!=GPU or os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE']!=str(args.cap):
            raise ValueError('Set original GPU/MPS environment before model load')
        with torch.inference_mode():
            rows,prepare,transfer,forward,first,finish=legacy.runtime(args,guard,identity)
            props=torch.cuda.get_device_properties(0)
            if props.multi_processor_count!=(108 if args.cap==100 else 54):raise ValueError('MPS cap ineffective')
            limit=float(subprocess.check_output(['nvidia-smi','--id='+GPU,'--query-gpu=power.limit','--format=csv,noheader,nounits'],text=True))
            if limit!=250:raise ValueError('Do not change original power cap')
            identity.update(visible_multiprocessors=props.multi_processor_count,power_limit_w=limit)
            accounting.write(args.output/'identity.json',identity)
            restored=wrap_tokenizers(prepare,tokenizer)
            if args.model in ('qwen','glm') and not restored:raise ValueError('Native tokenizer observation missing')
            raw=args.output/'telemetry_stdout.jsonl'
            sampler=ProcessTelemetrySampler(GPU,raw,command=observer.observer_command(GPU,raw))
            sampler.start();sampler.set_phase('original_serial_no_warmup');time.sleep(.15)
            for index in range(2048):
                guard.fast();sampler.check();row=rows[index%512];lo=len(tokenizer)
                a=time.perf_counter();cpu,positions=prepare([row]);b=time.perf_counter()
                inputs=transfer(cpu);torch.cuda.synchronize();c=time.perf_counter()
                call=time.perf_counter();values,wall,cuda=forward(inputs);returned=time.perf_counter()
                start=float(first());ready=start+float(wall)/1000
                if not a<=b<=c<=call<=start<ready<=returned+1e-5 or len(values)!=1 or not math.isfinite(values[0]):
                    raise ValueError('Invalid original runtime node timestamps')
                evidence=old.cpu_input_evidence(cpu)
                if isinstance(cpu,dict):ids=cpu.get('input_ids');mask=cpu.get('attention_mask')
                else:ids=mask=None
                records.append(dict(index=index,source_index=index%512,cycle=index//512,sample_id=row['sample_id'],
                    CPU_start=a,CPU_end=b,H2D_end=c,GPU_host_start=call,first_TF_host=start,
                    GPU_ready_host=ready,CPU_result_return=returned,diagnostic_first_TF_to_output_cuda_ms=cuda,
                    tokenizer_intervals=tokenizer[lo:],prediction=float(values[0]),frame_positions=positions[0],
                    input_ids=ids.tolist() if ids is not None else [[]],
                    attention_mask=mask.tolist() if mask is not None else [[]],input_evidence=evidence))
                del inputs,cpu
                if (index+1)%32==0:status('evaluating_original_strict_B1');print(f'{args.model} cap{args.cap} {index+1}/2048',flush=True)
            sampler.request('barrier');samples=sampler.stop();sampler=None
            save_samples(args.output/'telemetry.csv',samples)
            prediction=np.asarray([r['prediction'] for r in records],dtype=np.float32)
            if not all(np.array_equal(prediction[:512],prediction[i:i+512]) for i in (512,1024,1536)):
                raise ValueError('Four original prediction cycles differ')
            np.savez_compressed(args.output/'outputs.npz',prediction=prediction)
            intervals=accounting.retained_intervals(records)
            elapsed=sum(y-x for x,y in intervals)
            timing=dict(status='complete',model=args.model,video_calls=2048,unique_videos=512,cap=args.cap,
                corrected_GPU_ready_ms=elapsed*1000,time16_ms=elapsed*1000/128,equivalent_frame_ms=elapsed*1000/8192,
                records=records,default_original_forward=True,no_graph=True,no_prefetch=True,
                CPU_preparation_single_video=True,every_video_image_preparation_and_original_H2D_excluded=True,
                other_dispatch_sync_return_waits_retained=True)
            accounting.write(args.output/'timing.json',timing)
            report=accounting.summarize(timing,samples,913408);report.update(cap=args.cap,repeat=args.repeat)
            if args.model in ('clip','yolo'):report.update(native_effective_tokens=None,native_tokens_per_s=None,
                native_tokens_not_applicable='Original visual-only quality baseline; common workload denominator separately retained')
            accounting.write(args.output/'cost_report.json',report)
            finish();finish=None;guard.check()
            accounting.write(args.output/'input_files.json',[dict(sample_id=r['sample_id'],video_path=str(r['video_path']),
                bytes=Path(r['video_path']).stat().st_size,sha256=old.sha(Path(r['video_path']))) for r in rows[:512]])
            status('complete',time16_ms=report['time16_ms'],energy16_j=report['energy16_j'])
    except BaseException as error:
        status('failed',error=repr(error));accounting.write(args.output/'partial_records.json',records);raise
    finally:
        if sampler is not None:save_samples(args.output/'telemetry.csv',sampler.stop(raise_on_error=False))
        if finish is not None:finish()
        for owner,name,value in reversed(restored):setattr(owner,name,value)
        accounting.write(args.output/'identity.json',identity)
        guard.close(args.output/'process_audit.json')


def suite(args):
    import fcntl
    from LightGenV2.tasks.t06_video_quality_assessment import measure_mps_e2e_20261024 as old
    from run_temporal_remainder import process_snapshot
    args.root.mkdir();legacy=old.initialize_profile(args);active=False;child=None
    plan=[(m,c,r) for m in ('qwen','glm','clip','yolo') for c in (100,50) for r in (1,2,3)]
    retained=[]
    if len(args.retained_point)!=len(args.retained_cost_sha256):raise ValueError('Exact retained cost SHA required')
    for folder,digest in zip(args.retained_point,args.retained_cost_sha256):
        folder=folder.resolve()
        if folder.parent.parent!=TASK/'runs/simulation':raise ValueError('Canonical retained point only')
        previous=json.loads((folder.parent/'suite_status.json').read_text())
        point=json.loads((folder/'status.json').read_text());cost=json.loads((folder/'cost_report.json').read_text())
        if previous['status'] not in ('stopped','complete') or point['status']!='complete' or cost['status']!='complete':
            raise ValueError('Only fully complete retained points from a closed predecessor')
        if old.sha(folder/'cost_report.json')!=digest or cost['video_calls']!=2048:raise ValueError('Retained evidence changed')
        identity=json.loads((folder/'identity.json').read_text());now=sources()
        own=Path(__file__).relative_to(REPO).as_posix()
        if any(now.get(k)!=v for k,v in identity['sources'].items() if k!=own):raise ValueError('Original model/runtime source changed')
        name=f"{cost['model']}_cap{cost['cap']}_repeat{cost['repeat']:02d}"
        if folder.name!=name or name in {x['name'] for x in retained}:raise ValueError('Duplicate or mismatched retained identity')
        retained.append(dict(name=name,path=str(folder),cost_sha256=digest))
    names={x['name'] for x in retained}
    plan=[(m,c,r) for m,c,r in plan if f'{m}_cap{c}_repeat{r:02d}' not in names]
    state=dict(status='admission',pid=os.getpid(),deadline=args.deadline,plan=plan,completed=[],retained=retained,sources=sources())
    def save():accounting.write(args.root/'suite_status.json',state)
    save()
    try:
        with (TASK/'runs/simulation/.temporal_telemetry_a100.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            idle=0
            while idle<12:
                if time.time()>=args.deadline:raise TimeoutError('Finite admission queue deadline')
                snapshot=process_snapshot(GPU,-1,args.pipe)
                if snapshot['foreign_gpu_processes'] or snapshot['foreign_mps_clients']:
                    idle=0;state.update(status='waiting_foreign_process',admission_observation=snapshot);save()
                    time.sleep(5);continue
                try:legacy.validate_idle_memory(GPU)
                except RuntimeError as error:
                    idle=0;state.update(status='waiting_idle_memory',admission_observation=snapshot,
                        admission_memory_error=str(error));save();time.sleep(5);continue
                idle+=1
                time.sleep(.25)
            legacy.validate_idle_memory(GPU);legacy.engine.daemon(True);active=True
            for model,cap,repeat in plan:
                if time.time()>=args.deadline-300:raise TimeoutError('Preserve backup margin')
                mp,pt,python,receipt=legacy.lab.locations(model)
                name=f'{model}_cap{cap}_repeat{repeat:02d}';leaf=args.root/name
                command=[python,'-u',str(Path(__file__)),'--phase','worker','--root',str(args.root),
                    '--model',model,'--cap',str(cap),'--repeat',str(repeat),'--output',str(leaf),
                    '--model-path',str(mp),'--checkpoint',str(pt),'--deadline',str(args.deadline)]
                if receipt:command+=['--reproduction-receipt',str(receipt)]
                accounting.write(args.root/(name+'.command.json'),command)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,CUDA_MPS_PIPE_DIRECTORY=args.pipe,
                    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=str(cap),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',
                    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
                with (args.root/(name+'.log')).open('x') as log:
                    child=subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    state.update(status='running',phase=name,child=child.pid);save()
                    while child.poll() is None:
                        if time.time()>=args.deadline:raise TimeoutError('Finite measurement deadline')
                        time.sleep(1)
                if child.returncode!=0:raise RuntimeError('Preserve failed point; no blind retry')
                state['completed'].append(name);save()
            state.update(status='complete',finished=time.time())
    except BaseException as error:
        state.update(status='stopped',error=repr(error))
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGINT);child.wait(timeout=60)
        raise
    finally:
        try:
            if active:
                try:legacy.engine.daemon(False)
                except BaseException as error:
                    # The old cleanup also demanded the entire shared GPU be
                    # empty after quitting OUR private MPS daemon. A foreign
                    # process can make that postcondition false; preserve the
                    # cleanup error without losing the original stopped state.
                    state['cleanup_error']=repr(error)
        finally:save()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['suite','worker'],required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path);p.add_argument('--model',choices=['qwen','glm','clip','yolo'])
    p.add_argument('--model-path',type=Path);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--reproduction-receipt',type=Path);p.add_argument('--cap',type=int,choices=[50,100]);p.add_argument('--repeat',type=int)
    p.add_argument('--retained-point',type=Path,action='append',default=[])
    p.add_argument('--retained-cost-sha256',action='append',default=[])
    p.add_argument('--deadline',type=float,required=True);args=p.parse_args()
    args.root=args.root.resolve();args.suite_root=args.root;args.profile='lab';args.gpu_uuid=GPU
    args.pipe='/DATA/DATA1/guest3/.cache/'+args.root.name
    args.manifest=REPO/'experiments/lgvq_four_stage_optical_electronic_109_no_attention_vqa/artifacts/lgvq_train2250_test558.csv'
    args.batch=1;args.precision='native'
    if args.root.parent!=TASK/'runs/simulation' or not args.root.name.startswith('a100_lab_corrected_energy_20261026_'):
        raise ValueError('New canonical lab root required')
    if not 0<args.deadline-time.time()<=36000:raise ValueError('Finite continuation required')
    if args.phase=='suite':suite(args)
    else:
        args.output=args.output.resolve()
        if args.output.parent!=args.root:raise ValueError('Wrong leaf')
        worker(args)


if __name__=='__main__':main()
