"""Original A10040 profile with strict per-forward TF boundaries and waits.

Frozen-runtime routing is unchanged; timing is an explicitly new profile.
No global compute-mode/power change, training, Graph or protected B1 overwrite.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import time
from types import SimpleNamespace
from . import rental_busy_resource_20261020 as engine
from . import rental_precision_recheck as shared
from . import lab_fp32_recheck as fp32
from . import measure_selected_execution as selected
from . import measure_unified_inference_recheck as previous
from . import evaluate_temporal_baseline_batches as frozen
from . import unified_resident_utilization as uniform
from . import run_temporal_telemetry_suite as catalog
from . import strict_tf_window_20261020 as strict

TASK=shared.TASK;REPO=shared.REPO
ROOT=TASK/'runs/simulation/lgvq_busy_resource_lab_20261020'/strict.PROTOCOL
PIPE='/DATA/DATA1/guest3/.cache/t06_mps_strict_tf_20261020'
PREVIOUS_CONTINUATION='calibration_1140'
CONTINUATION='duration_1155'
GLM_RECOVERY='glm_sampling_recovery01'


def point_plan(session):
    """A finite admission plan, not a change to the GPU measurement loop."""
    if session==GLM_RECOVERY:
        return [('fp32','glm','mps',100),('fp32','glm','mps',50),('fp32','glm','temporal',100)]
    return [(precision,model,method,cap) for precision in ('native','fp32')
            for model in ('clip','yolo','qwen','glm')
            for method,cap in (('mps',100),('mps',50),('temporal',100))]


def completed_reference(name):
    """Only the three completed predecessor points may be reused, in place."""
    current=ROOT/name
    if (current/'status.json').exists():return current
    if ROOT.name==CONTINUATION and name in ('mps_clip_native_100','mps_clip_native_50','temporal_clip_native_100'):
        session=PREVIOUS_CONTINUATION if name=='temporal_clip_native_100' else 'recovery_1102'
        prior=ROOT.parent/session/name
        if json.loads((prior/'status.json').read_text())['status']!='complete':
            raise RuntimeError('Completed predecessor changed')
        return prior
    return current


def sources():
    return {**engine.sources(),**selected.sources(),**uniform.sources(),
        **{str(p.relative_to(REPO)):frozen.sha256(p) for p in (Path(__file__),Path(fp32.__file__),Path(previous.__file__),Path(strict.__file__))}}


def locations(model):
    if model=='glm':
        original=TASK/'runs/simulation'/uniform.SPECIAL['glm'][0]
        command=json.loads((original/'a100_batch01/report.json').read_text())['identity']['command']
        return Path(command[command.index('--model')+1]),original/'train/best_checkpoint.pt',uniform.SPECIAL['glm'][2],None
    mp=catalog.QWEN_MODEL if model=='qwen' else catalog.reproduction.MODEL_PATHS[model]
    pt=catalog.QWEN_PT if model=='qwen' else catalog.PREVIOUS/model/'checkpoints/temporal/best_checkpoint.pt'
    python=catalog.QWEN_ENV if model=='qwen' else catalog.reproduction.ENV_PATHS[model]
    receipt=None if model=='qwen' else catalog.PREVIOUS/model/'reproduction_receipt.json'
    return mp,pt,python,receipt


def native_runtime(args,guard,identity):
    import torch
    from . import quality_token_common as core
    a=SimpleNamespace(**vars(args));a.baseline=args.model;a.model=args.model_path
    if args.model=='qwen':
        model,head,rows,prepare,transfer,native,build,finish,assets=selected.qwen_runtime(a)
        cpu,positions=prepare(rows[:args.batch]);inputs=transfer(cpu)
        if args.batch==1:compute=native;first=model.model.visual.blocks[0]
        else:adapter,compute=build(cpu,inputs);first=adapter.visual.blocks[0]
    elif args.model=='glm':
        model,head,prepare,transfer,compute,reference,finish=uniform.special_runtime(a)
        rows=[r for r in core.read_manifest(args.manifest) if r['split']=='test']
        cpu,positions=prepare(rows[:args.batch]);inputs=transfer(cpu)
        first=model.model.visual.blocks[0];assets=previous.costs.model_hashes(args.model_path)
    else:
        assets=previous.costs.model_hashes(args.model_path)
        model,head,rows,prepare,transfer,native,timer,handles=frozen.load_runtime(a);timer.close()
        compute=uniform.execution.make_gpu_forward(a,model,head,native)
        cpu,positions=prepare(rows[:args.batch]);inputs=transfer(cpu)
        first=model.visual.transformer.resblocks[0] if args.model=='clip' else model.model[0]
        def finish():
            if previous.costs.model_hashes(args.model_path)!=assets:raise RuntimeError('Frozen assets changed')
            for handle in handles:handle.remove()
    if frozen.sha256(args.checkpoint)!=previous.costs.CHECKPOINTS[args.model] or len(rows)!=558:
        raise RuntimeError('Original frozen head and558 TEST required')
    guard.bind()
    identity.update(checkpoint_sha256=frozen.sha256(args.checkpoint),assets=assets,
        sample_ids=[r['sample_id'] for r in rows[:args.batch]],frame_positions=positions,
        floating_parameter_dtypes=sorted({str(p.dtype) for p in model.parameters() if p.is_floating_point()}))
    def measured(x):
        with torch.inference_mode():return compute(x),None
    return model,inputs,first,measured,finish


def worker(args):
    import torch
    # Explicit platform configuration of the shared engine, not a model patch.
    engine.GPU=frozen.A100_UUID;engine.PIPE=PIPE;engine.ROOT=ROOT
    args.gpu_uuid=frozen.A100_UUID;args.power_limit_w=250
    if args.output.exists():raise RuntimeError('Preserve prior point')
    if args.model!='glm':
        pre=SimpleNamespace(**vars(args));pre.baseline=args.model;pre.model=args.model_path
        frozen.preflight_files(pre)
    args.output.mkdir(parents=True);bound=sources()
    parent=engine.MPSGuard if args.method=='mps' else shared.Guard
    class Guard(parent):
        def check(self):
            if sources()!=bound:raise RuntimeError('Lab bound source changed')
            super().check()
    guard=Guard(args);finish=None
    identity=dict(model=args.model,precision=args.precision,method=args.method,batch=args.batch,cap=args.cap,
        gpu_uuid=frozen.A100_UUID,pid=os.getpid(),sources=bound,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
    try:
        guard.check();torch.set_num_threads(8)
        power=float(subprocess.check_output(['nvidia-smi','--id='+frozen.A100_UUID,'--query-gpu=power.limit','--format=csv,noheader,nounits'],text=True))
        if abs(power-250)>.1:raise RuntimeError('Unchanged250W required')
        if args.method=='mps' and os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE')!=str(args.cap):raise RuntimeError('Cap before CUDA context required')
        if args.precision=='fp32':
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
            model,rows,prepare,transfer,compute,first,assets=fp32.runtime(args,guard,identity)
            cpu,positions=prepare(rows[:args.batch]);inputs=transfer(cpu)
            identity.update(sample_ids=[r['sample_id'] for r in rows[:args.batch]],frame_positions=positions)
            finish=lambda:None
        else:model,inputs,first,compute,finish=native_runtime(args,guard,identity)
        identity['visible_multiprocessors']=torch.cuda.get_device_properties(0).multi_processor_count
        if args.method=='mps' and args.cap==50:
            full=json.loads((completed_reference(f'mps_{args.model}_{args.precision}_100')/'identity.json').read_text())['visible_multiprocessors']
            identity['visible_sm_ratio_to100']=identity['visible_multiprocessors']/full
            if not .4<=identity['visible_sm_ratio_to100']<=.55:raise RuntimeError('Resource cap audit failed')
        shared.old.write(args.output/'identity.json',identity)
        if args.method=='mps':conditions=[('mps_cap'+str(args.cap),1)]
        else:
            # Each paced condition calibrates its OWN individually synchronized
            # execution. Old queue8 duties do not describe the new protocol.
            conditions=[('unrestricted',1),('busy50',.5)]
        identity['timing_protocol']=strict.PROTOCOL
        args.validate_pacing=args.session in (PREVIOUS_CONTINUATION,CONTINUATION,GLM_RECOVERY)
        args.minimum_charged_s=35. if args.session in (CONTINUATION,GLM_RECOVERY) else 0.
        identity['pacing_calibration']='full_window_validation_v1' if args.validate_pacing else 'short_feedback60s_v1'
        identity['minimum_charged_s']=args.minimum_charged_s
        for label,duty in conditions:
            report=strict.measure(args,compute,inputs,first,args.output/label,guard,duty)
            if report['target50_reached'] is False:
                raise RuntimeError('Three complete trials miss50+/-5; preserve all, no automatic retry')
        if previous.costs.model_hashes(args.model_path)!=identity['assets']:raise RuntimeError('Assets changed')
        finish();guard.check();shared.old.write(args.output/'status.json',dict(status='complete',time=time.time()))
    except BaseException as error:
        shared.old.write(args.output/'status.json',dict(status='failed',error=repr(error),time=time.time()));raise
    finally:shared.old.write(args.output/'identity.json',identity);guard.close(args.output/'process_audit.json')


def suite(args):
    import fcntl
    with (TASK/'runs/simulation/.temporal_telemetry_a100.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if ROOT.exists():raise RuntimeError('Preserve prior suite')
        ROOT.mkdir();state=dict(status='waiting_for_idle',deadline=args.deadline,completed=[],pid=os.getpid())
        def save():shared.old.write(ROOT/'suite_status.json',state)
        stable=0
        while stable<3:
            if time.time()>=args.deadline-180:state['status']='waiting_expired';save();return
            records=shared.guard_records(frozen.A100_UUID);stable=stable+1 if not records else 0
            state.update(foreign=records,stable=stable,time=time.time());save()
            if stable<3:time.sleep(10)
        engine.GPU=frozen.A100_UUID;engine.PIPE=PIPE;engine.ROOT=ROOT;active=False
        try:
            for precision in ('native','fp32'):
                for model in ('clip','yolo','qwen','glm'):
                    for method,cap in (('mps',100),('mps',50),('temporal',100)):
                        if (precision,model,method,cap) not in point_plan(args.session):continue
                        if args.deadline-time.time()<180:raise TimeoutError('Finite current window, no new point')
                        if active and method!='mps':engine.daemon(False);active=False
                        if not active and method=='mps':
                            if shared.guard_records(frozen.A100_UUID):raise RuntimeError('Foreign GPU process')
                            engine.daemon(True);active=True
                        batch=1 if method=='mps' else {'qwen':3,'glm':2 if precision=='native' else 1,'clip':10,'yolo':3}[model]
                        name=f'{method}_{model}_{precision}_{cap}';output=ROOT/name
                        if args.session==CONTINUATION and name in ('mps_clip_native_100','mps_clip_native_50','temporal_clip_native_100'):
                            prior=completed_reference(name)
                            condition='mps_cap'+str(cap) if method=='mps' else 'busy50'
                            report=json.loads((prior/condition/'report.json').read_text())
                            identity=json.loads((prior/'identity.json').read_text())
                            if report['status']!='complete' or report['batch']!=(1 if method=='mps' else 10) or len(report['repeats'])!=3:
                                raise RuntimeError('Invalid completed predecessor')
                            if method=='temporal' and report['target50_reached'] is not True:raise RuntimeError('Prior control must qualify')
                            if frozen.sha256(locations(model)[1])!=identity['checkpoint_sha256'] or previous.costs.model_hashes(locations(model)[0])!=identity['assets']:
                                raise RuntimeError('Reused frozen assets changed')
                            state.setdefault('reused',{})[name]=dict(path=str(prior.relative_to(REPO)),
                                report_sha256=frozen.sha256(prior/condition/'report.json'),
                                identity_sha256=frozen.sha256(prior/'identity.json'))
                            save();continue
                        _,_,python,_=locations(model)
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES=frozen.A100_UUID,OMP_NUM_THREADS='8')
                        if method=='mps':env.update(CUDA_MPS_PIPE_DIRECTORY=PIPE,CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=str(cap))
                        else:env.pop('CUDA_MPS_PIPE_DIRECTORY',None);env.pop('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE',None)
                        command=[python,'-u','-m',__name__,'--model',model,'--precision',precision,'--method',method,
                            '--cap',str(cap),'--batch',str(batch),'--deadline',str(args.deadline),'--manifest',str(args.manifest),'--output',str(output)]
                        # __name__ is __main__ in a CLI execution, so use the package-qualified entry.
                        command[3]=__package__+'.lab_busy_resource_20261020'
                        command.extend(['--session',args.session])
                        shared.old.write(ROOT/(name+'.command.json'),command)
                        with (ROOT/(name+'.log')).open('x') as log:
                            child=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                            state.update(status='running',phase=name,child_pid=child.pid);save()
                            try:code=child.wait(timeout=max(1,args.deadline-time.time()))
                            except BaseException:
                                os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=15);raise
                        if code:raise RuntimeError(f'{name} failed exit{code}; preserve evidence, no automatic retry')
                        state['completed'].append(name);save()
            state.update(status='complete',finished=time.time())
        except BaseException as error:state.update(status='stopped',error=repr(error));raise
        finally:
            if active:
                try:engine.daemon(False);state['mps_active']=False
                except BaseException as error:state['mps_shutdown_error']=repr(error)
            save()


def main():
    global ROOT,PIPE
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',action='store_true')
    p.add_argument('--session',choices=('original','idle_1049','recovery_1102',PREVIOUS_CONTINUATION,CONTINUATION,GLM_RECOVERY),default='original')
    p.add_argument('--model',choices=('qwen','glm','clip','yolo'))
    p.add_argument('--method',choices=('temporal','mps'))
    p.add_argument('--precision',choices=('native','fp32'))
    p.add_argument('--cap',type=int,choices=(50,100),default=100)
    p.add_argument('--batch',type=int);p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,default=ROOT)
    args=p.parse_args()
    if args.session!='original':
        original=json.loads((ROOT/'suite_status.json').read_text())
        if original.get('completed') or list(ROOT.glob('*/identity.json')):
            raise RuntimeError('New admission only after zero-measurement expired predecessor')
        if args.session=='recovery_1102':
            failed=ROOT/'idle_1049'
            state=json.loads((failed/'suite_status.json').read_text())
            if state['status']!='stopped' or state['completed'] or list(failed.glob('*/*/report.json')):
                raise RuntimeError('One authorized recovery only; preserve completed points')
            if Path('/proc/'+str(state['pid'])).exists():raise RuntimeError('Prior suite still alive')
        if args.session in (PREVIOUS_CONTINUATION,CONTINUATION):
            prior=ROOT/'recovery_1102';state=json.loads((prior/'suite_status.json').read_text())
            if state['status']!='stopped' or state['completed']!=['mps_clip_native_100','mps_clip_native_50']:
                raise RuntimeError('Exact preserved predecessor required')
            if Path('/proc/'+str(state['pid'])).exists():raise RuntimeError('Prior suite alive')
            failed=json.loads((prior/'temporal_clip_native_100/status.json').read_text())
            if 'miss50+/-5' not in failed.get('error',''):raise RuntimeError('Expected controller-target failure only')
        if args.session==CONTINUATION:
            prior=ROOT/PREVIOUS_CONTINUATION;state=json.loads((prior/'suite_status.json').read_text())
            if state['status']!='stopped' or state['completed']!=['temporal_clip_native_100']:
                raise RuntimeError('Exact completed calibration predecessor required')
            if Path('/proc/'+str(state['pid'])).exists():raise RuntimeError('Prior suite alive')
            failed=json.loads((prior/'mps_yolo_native_100/status.json').read_text())
            if 'shorter than30s' not in failed.get('error',''):raise RuntimeError('Expected zero-complete-trial duration failure only')
        if args.session==GLM_RECOVERY:
            prior=ROOT/CONTINUATION;state=json.loads((prior/'suite_status.json').read_text())
            accuracy=json.loads((prior/'accuracy_fp32/suite_status.json').read_text())
            failed=json.loads((prior/'mps_glm_fp32_100/status.json').read_text())
            if (state['status']!='stopped' or accuracy['status']!='complete'
                    or 'Power observation older than250ms' not in failed.get('error','')
                    or any(Path('/proc/'+str(s['pid'])).exists() for s in (state,accuracy))):
                raise RuntimeError('Sampling failure or prior accuracy still active')
            expected=['clip_batch01','clip_batch10','yolo_batch01','yolo_batch03','qwen_batch01','qwen_batch03']
            if accuracy['completed']!=expected:raise RuntimeError('Preserve all six completed accuracy references')
        ROOT=ROOT/args.session;PIPE=PIPE+'_'+args.session
        if args.output==p.get_default('output'):args.output=ROOT
    args.output=args.output.resolve()
    if not args.output.is_relative_to(ROOT) or os.environ.get('CUDA_VISIBLE_DEVICES')!=frozen.A100_UUID or not 0<args.deadline-time.time()<=7200:raise ValueError('Original physical A100 and finite canonical run')
    if args.suite:return suite(args)
    if args.method=='mps' and args.batch!=1:raise ValueError('MPS B1 fixed')
    expected={'qwen':3,'glm':2 if args.precision=='native' else 1,'clip':10,'yolo':3}[args.model]
    if args.method=='temporal' and args.batch!=expected:raise ValueError('Fixed existing/capacity-declared batch')
    args.model_path,args.checkpoint,_,args.reproduction_receipt=locations(args.model)
    worker(args)


if __name__=='__main__':main()
