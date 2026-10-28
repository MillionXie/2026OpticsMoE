"""Fresh three-warm-field LG kernel trace, three independent 512-video rounds.

Full original predictions and FFT execution are retained. Pure kernel durations
feed a separately labelled fixed-optics resource-schedule estimate. No claimed
physical photonic stopwatch or measured hybrid-deployment power.
"""
import argparse
import ast
import functools
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ENTRY=Path(__file__).resolve();HERE=ENTRY.parent;REPO=HERE.parents[3]
TASK=REPO/'LightGenV2/tasks/t06_video_quality_assessment';SIM=TASK/'runs/simulation'
GPU='GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55'
PYTHON='/home/guest3/miniconda3/envs/xml/bin/python'
sys.path.insert(0,str(HERE));sys.path.insert(0,str(REPO))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=True,indent=2,allow_nan=False)+'\n',encoding='utf8')
    temporary.replace(path)

def sources():
    import lgvq_async_worker_20261026 as old
    paths=[ENTRY,HERE/'lgvq_compute_scope_20261029.py',HERE/'lgvq_a100_patch_parallel_20261027.py',
           HERE/'run_corrected_boundaries_20261026.py',HERE/'qwen_shared_serial_20261026.py',
           HERE/'qwen_strict_serial_20261026.py',HERE/'run_temporal_remainder.py',
           HERE/'first_tf_supplement_20261025.py',HERE/'lab_corrected_energy_triples_20261026.py',
           TASK/'rental_busy_resource_20261020.py']
    return {**old.source_map(),**{p.relative_to(REPO).as_posix():sha(p) for p in paths}}

def verify(commit,frozen):
    if not re.fullmatch('[a-f0-9]{40}',commit):raise ValueError('Published commit required')
    for name,digest in frozen.items():
        b=subprocess.check_output(['git','show',commit+':'+name],cwd=REPO,timeout=10)
        if hashlib.sha256(b).hexdigest()!=digest:raise ValueError('Unpublished source: '+name)

def control(pipe,command):
    return subprocess.run(['nvidia-cuda-mps-control'],input=command+'\n',text=True,capture_output=True,
        env=dict(os.environ,CUDA_MPS_PIPE_DIRECTORY=pipe),timeout=5)

def alive(pid):
    try:return Path('/proc',str(int(pid)),'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False

def execute(args,guard):
    from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
    import multiprocessing
    from types import MethodType
    import numpy as np
    import torch
    import lgvq_compute_scope_20261029 as scope
    import lgvq_live_frontend as live
    import lgvq_gpu_pipeline_20261026 as base
    import lgvq_physical_projection_20261026 as projection
    import lgvq_scoreonly_20261026 as score
    import lgvq_hoisted_pipeline_20261026 as partition
    import lgvq_a100_patch_parallel_20261027 as convolution
    import lgvq_async_worker_20261026 as old
    from lgvq_pipeline_demo_20261025 import PreparedFront
    from LightGenV2.tasks.t06_video_quality_assessment import lab_runtime as native
    from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.data import read_manifest
    from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54 import cache_qwen_front as text
    from LightGenV2.tasks.t06_video_quality_assessment import e2e_telemetry_observer as observer
    from LightGenV2.tasks.t06_video_quality_assessment.nvml_observer_process import ProcessTelemetrySampler
    from LightGenV2.tasks.t06_video_quality_assessment.batch_telemetry import save_samples
    from LightGenV2.tasks.t06_video_quality_assessment.strict_tf_window_20261020 import interval_energy
    output=args.output;sampler=None;restore=None
    def status(value,**kw):write(output/'status.json',dict(status=value,pid=os.getpid(),unix=time.time(),**kw))
    checkpoint=TASK/'runs/simulation/multivideo16x4_rank_s163/best_checkpoint.pt'
    reference=REPO/old.REFERENCE_RELATIVE
    if sha(reference)!=old.REFERENCE_SHA256 or sha(checkpoint)!=live.PINS['temporal'][1]:
        raise ValueError('Frozen weights/reference changed')
    with np.load(reference,allow_pickle=False) as f:expected=f['prediction'][:32].copy()
    if expected.shape!=(32,16) or expected.dtype!=np.float32:raise ValueError('Exact ordered reference required')
    torch.set_num_threads(1)
    front=live.LiveFront('cuda:0');model,_=native.load_model('temporal',checkpoint,'cuda:0');model.requires_grad_(False)
    guard.bind()
    if torch.cuda.get_device_properties(0).multi_processor_count!=108:raise ValueError('Actual LG100 allocation required')
    position=PreparedFront(front).position4.repeat(16,1).to(dtype=torch.bfloat16)
    conv_receipt=convolution.install(front,partitions=2)
    rows=[dict(sample_id=r.sample_id,video_path=str(r.video_path)) for r in read_manifest(live.MANIFEST) if r.split=='test'][:512]
    if sha(live.MANIFEST)!=live.MANIFEST_SHA or len({r['sample_id'] for r in rows})!=512:raise ValueError('Original ordered TEST inputs required')
    status('preparing_raw_normalized_pixels_not_neural_features',completed=0)
    prepared=[]
    with ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn'),
            initializer=scope.prepare_cpu_worker,initargs=(str(front.model_path),)) as pool:
        pending=[pool.submit(scope.prepare_video,row) for row in rows]
        for i,future in enumerate(pending):
            prepared.append(future.result(timeout=150));guard.fast()
            if (i+1)%32==0:status('preparing_raw_normalized_pixels_not_neural_features',completed=i+1)
    receipts=[p['processor_receipt'] for p in prepared]
    if any(r!=receipts[0] for r in receipts):raise ValueError('Processor implementation changed')
    from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
    _,patch_fn,patch_receipt=scope.processor_adapter(Qwen2VLImageProcessorFast._preprocess,lambda *a:None)
    if patch_receipt!=receipts[0]:raise ValueError('Main/CPU worker patch AST differ')
    write(output/'input_preparation.json',dict(rows=[dict(**r,sha256=sha(r['video_path']),
        original_flat_sha256=p['original_flat_sha256'],original_flat_shape=p['original_flat_shape'],
        external_preparation=p['external_preparation']) for r,p in zip(rows,prepared)],processor=patch_receipt,
        excluded='decode, resize, normalization and quality-pixel preparation; no learned features cached'))
    meter=scope.ScopeMeter(torch)
    original=type(model).forward
    score_tree,score_receipt=score.score_forward_AST(inspect.getsource(original))
    tree,partition_receipt=partition.partition_forward(ast.unparse(score_tree))
    namespace=dict(original.__globals__)
    router_tree,router_receipt=score.router_score_AST(inspect.getsource(namespace['_standardized_router']))
    exec(compile(router_tree,'<original_score_router>','exec'),namespace)
    exec(compile(tree,'<original_partition>','exec'),namespace)
    prefix_fn=namespace['prepare_native_prefix'];tail_fn=namespace['forward_from_prefix']
    tail_source=ast.unparse(next(n for n in tree.body if n.name=='forward_from_prefix'))
    tail_fn,tail_sha=scope.annotate_tail(tail_fn,meter,tail_source)
    restore,boundary_receipt=projection.install_simulation_boundaries(model,meter)
    fusion_original=[]
    for i,owner in enumerate(model.fusions):
        native_fusion=owner.forward;fusion_original.append((owner,native_fusion))
        def wrap(original,label):
            @functools.wraps(original)
            def call(*a,**kw):
                with meter.excluded(label,'fusion'):return original(*a,**kw)
            return call
        owner.forward=wrap(native_fusion,str(i))
    native_fields=model.serial_router.fields
    def fields(*a,**kw):
        with meter.scope('adapter_language'):return native_fields(*a,**kw)
    model.serial_router.fields=fields
    patch_rows=[];token_rows=[];outputs=[];warm_outputs=[];hosts=[]
    fs=torch.cuda.current_stream();ts=torch.cuda.Stream()
    write(output/'compute_identity.json',dict(profile=scope.PROFILE,fields=32,videos_per_field=16,
        frames_per_field=64,unique_videos=512,explicit_extra_warm_fields=3,cap=100,visible_SM=108,
        power_limit_W=250,model_sha256=sha(checkpoint),reference_sha256=sha(reference),
        sources=sources(),processor=patch_receipt,convolution=conv_receipt,score=score_receipt,
        partition=partition_receipt,router=router_receipt,tail_AST_sha256=tail_sha,boundaries=boundary_receipt,
        physical_optical_hardware_measured=False,energy_qualified=False,
        resource_schedule_is_separate_from_actual_FFT_software_wall=True))
    def field_run(index,warm,pool):
        guard.fast();meter.field=-index-1 if warm else index;meter.pass_count=0;meter.phase='adapter_visual'
        start=time.perf_counter();lo=len(meter.events)
        futures=[pool.submit(scope.patch_video,p,meter.field,slot,patch_fn)
                 for slot,p in enumerate(prepared[index*16:(index+1)*16])]
        a=time.perf_counter();ca=time.thread_time()
        rendered=text.render_prompt(front.tokenizer,target_name='temporal',prompt=text.TARGET_PROMPTS['temporal'])
        tok=front.tokenizer([rendered],padding=True,return_tensors='pt',add_special_tokens=False)
        cb=time.thread_time();b=time.perf_counter()
        tokens=dict(field=meter.field,host_start=a,host_end=b,thread_cpu_s=cb-ca,
                    ids=tok['input_ids'].tolist(),mask=tok['attention_mask'].tolist())
        packed=[future.result(timeout=30) for future in futures]
        with meter.scope('excluded_input_transfer'):
            pixels=torch.cat([p[0] for p in packed]).to('cuda',dtype=torch.bfloat16)
            quality=torch.cat([torch.from_numpy(p['quality']) for p in prepared[index*16:(index+1)*16]]).unsqueeze(0).to('cuda')
            current=dict(pixels=pixels,quality=quality,ids=tok['input_ids'].to('cuda'),mask=tok['attention_mask'].bool().to('cuda'))
        meter.open_electronics()
        try:
            batch=base.native_lg_front(front,position,current,meter,ts)
            prefix=prefix_fn(model,**batch,optical_enabled=True)
            result=tail_fn(model,prefix,optical_enabled=True)
        finally:meter.close_electronics()
        with meter.scope('excluded_score_transfer'):
            value=result['prediction'].float().cpu().numpy().copy()
        torch.cuda.synchronize();end=time.perf_counter()
        if meter.pass_count!=6:raise ValueError('Exactly six complete coherent fields per group required')
        if warm:warm_outputs.append(value)
        else:
            outputs.append(value);patch_rows.extend(p[1] for p in packed);token_rows.append(tokens)
        hosts.append(dict(field=meter.field,warm=warm,host_start=start,host_end=end,event_indices=[lo,len(meter.events)]))
        status('three_extra_warm_fields' if warm else 'recording_formal_operator_trace',completed=index+1,total=3 if warm else 32)
    try:
        sampler=ProcessTelemetrySampler(GPU,output/'telemetry_stdout.jsonl',command=observer.observer_command(GPU,output/'telemetry_stdout.jsonl'))
        sampler.start();sampler.set_phase('LG_FFT_execution_diagnostic_not_photonic_deployment')
        until=time.monotonic()+6
        while len(sampler.samples)<4:
            sampler.check();guard.fast()
            if time.monotonic()>until:raise RuntimeError('Original initial telemetry missing')
            time.sleep(.01)
        with ThreadPoolExecutor(max_workers=16) as pool,torch.inference_mode():
            for field in range(3):field_run(field,True,pool)
            # All three real warm fields are complete before the formal trace.
            mono_start=time.monotonic();host_start=time.perf_counter()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=False,profile_memory=False,with_stack=False) as profiler:
                for field in range(32):sampler.check();field_run(field,False,pool)
            host_end=time.perf_counter();mono_end=time.monotonic()
        # Retain all scored outputs and raw timing before numerical admission.
        np.savez_compressed(output/'outputs.npz',prediction=np.concatenate(outputs),warm_prediction=np.concatenate(warm_outputs))
        events=[{k:v for k,v in e.items() if k not in ('start','end')}|dict(cuda_elapsed_ms=e['start'].elapsed_time(e['end'])) for e in meter.events]
        write(output/'raw_operator_records.json',dict(host_records=hosts,patch=patch_rows,tokenizer=token_rows,
            enclosing_event_spans_diagnostic_not_pure_kernel_time=events,formal_host_start=host_start,
            formal_host_end=host_end,formal_monotonic_start=mono_start,formal_monotonic_end=mono_end,
            actual_execution_is_FFT_software=True,all_raw_scheduling_waits_retained_but_not_charged=True))
        profiler.export_chrome_trace(str(output/'raw_kernel_trace.json'))
        sampler.request('barrier');samples=sampler.stop();sampler=None;save_samples(output/'telemetry.csv',samples)
        array=np.concatenate(outputs);warm=np.concatenate(warm_outputs)
        exact=array.shape==expected.shape and array.dtype==expected.dtype and array.tobytes()==expected.tobytes()
        warm_exact=warm.shape==(3,16) and warm.tobytes()==expected[:3].tobytes()
        write(output/'numeric_audit.json',dict(all_formal_scores_bitwise_equal=exact,warm_scores_bitwise_equal=warm_exact,
            equal_scores=int(np.count_nonzero(array.view('uint32')==expected.view('uint32'))),
            maximum_absolute_difference=float(np.max(np.abs(array-expected))),reference_sha256=sha(reference)))
        if not exact or not warm_exact:raise ValueError('Original exact same-A100 per-video output gate failed')
        kernels=scope.traced_kernels(json.loads((output/'raw_kernel_trace.json').read_text()))
        write(output/'kernel_phase_records.json',kernels)
        estimate=scope.summarize_compute(kernels,patch_rows,token_rows,32)
        estimate.update(all_predictions_bitwise_equal=True,raw_trace_sha256=sha(output/'raw_kernel_trace.json'),
            raw_operator_records_sha256=sha(output/'raw_operator_records.json'),outputs_sha256=sha(output/'outputs.npz'),
            actual_FFT_software_wall_s=host_end-host_start,
            diagnostic_FFT_board_energy=interval_energy([(mono_start,mono_end)],samples,host_w=0.))
        write(output/'compute_report.json',estimate);guard.check()
        return estimate
    finally:
        meter.close_electronics()
        if restore:restore()
        for owner,original_fusion in fusion_original:owner.forward=original_fusion
        model.serial_router.fields=native_fields
        if sampler is not None:save_samples(output/'telemetry.csv',sampler.stop(raise_on_error=False))

def worker(args):
    import run_corrected_boundaries_20261026 as original
    import qwen_shared_serial_20261026 as shared
    frozen=json.loads((args.root/'source_receipt.json').read_text())
    if sha(args.root/'source_receipt.json')!=args.source_receipt_sha256 or sources()!=frozen['sources']:
        raise ValueError('Frozen runtime source changed')
    verify(frozen['published_commit'],frozen['sources'])
    shared.base.GPU=GPU;shared.base.PIPE=args.pipe;original.base.GPU=GPU;original.base.PIPE=args.pipe
    args.output.mkdir(exist_ok=False);guard=None
    write(args.output/'identity.json',dict(pid=os.getpid(),private_MPS_pipe=args.pipe,cap=100,published_commit=frozen['published_commit'],sources=frozen['sources']))
    try:
        write(args.output/'status.json',dict(status='loading_original_model',pid=os.getpid()))
        guard=original.ExclusiveGuard(args.deadline,100)
        report=execute(args,guard)
        if sources()!=frozen['sources']:raise ValueError('Runtime sources changed during run')
        write(args.output/'status.json',dict(status='complete',pid=os.getpid(),time16_s=report['time16_s']))
    except BaseException as error:
        write(args.output/'status.json',dict(status='failed',pid=os.getpid(),error=repr(error)));raise
    finally:
        if guard:guard.close(args.output/'process_audit.json')

def suite(args):
    import fcntl
    import qwen_shared_serial_20261026 as shared
    import lab_corrected_energy_triples_20261026 as admission
    from first_tf_supplement_20261025 import serialize_mps_commands
    from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
    serialize_mps_commands()
    predecessor=json.loads((args.after_suite/'suite_status.json').read_text())
    if (predecessor['status']!='complete' or not predecessor['mps_released'] or alive(predecessor['pid'])
            or predecessor['completed']!=[f'qwen_cap50_repeat{i:02d}' for i in (1,2,3)]
            or control('/DATA/DATA1/guest3/.cache/'+args.after_suite.name,'get_server_list').returncode==0):
        raise ValueError('Completed and truly exited new Qwen triple required')
    frozen=sources();commit=os.environ['MEASUREMENT_PUBLISHED_COMMIT'];verify(commit,frozen)
    args.root.mkdir(exist_ok=False)
    write(args.root/'source_receipt.json',dict(published_commit=commit,sources=frozen))
    receipt_sha=sha(args.root/'source_receipt.json')
    state=dict(status='admission',pid=os.getpid(),deadline=args.deadline,completed=[],
               plan=[f'lg_repeat{i:02d}' for i in (1,2,3)],after_suite=str(args.after_suite))
    lock=(SIM/'.temporal_telemetry_a100.lock').open('a');child=None;active=False
    def bounded():
        if time.time()>=args.deadline:raise TimeoutError('Finite LG suite deadline')
        if sources()!=frozen:raise ValueError('Frozen source changed')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        idle=0
        while idle<12:
            bounded();snapshot=admission.admission_snapshot(args.pipe)
            memory=float(subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=5))
            if not math.isfinite(memory) or memory<0:raise ValueError('Bad GPU memory')
            if snapshot['foreign_gpu_processes'] or snapshot['foreign_mps_clients'] or memory>=384:
                idle=0;state.update(status='waiting_foreign_or_memory',snapshot=snapshot,memory_mib=memory)
                write(args.root/'suite_status.json',state);time.sleep(5);continue
            idle+=1;time.sleep(.25)
        if control(args.pipe,'get_server_list').returncode==0:raise ValueError('Unexpected private daemon')
        engine.GPU=GPU;engine.PIPE=args.pipe;engine.ROOT=args.root;engine.daemon(True);active=True
        shared.base.GPU=GPU;shared.base.PIPE=args.pipe
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_MPS_PIPE_DIRECTORY=args.pipe,
            CUDA_MPS_ACTIVE_THREAD_PERCENTAGE='100',CUDA_MODULE_LOADING='LAZY',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
        for name in state['plan']:
            bounded();leaf=args.root/name
            command=[PYTHON,'-u',str(ENTRY),'--phase','worker','--root',str(args.root),'--output',str(leaf),
                     '--deadline',str(args.deadline),'--source-receipt-sha256',receipt_sha]
            write(args.root/(name+'.command.json'),command)
            with (args.root/(name+'.log')).open('x') as log:
                child=subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state.update(status='running',child_pid=child.pid,phase=name);write(args.root/'suite_status.json',state)
                while child.poll() is None:bounded();time.sleep(1)
            if child.returncode:raise RuntimeError('Failed LG leaf preserved; no blind retry')
            if json.loads((leaf/'status.json').read_text())['status']!='complete':raise ValueError('Incomplete LG result')
            state['completed'].append(name);write(args.root/'suite_status.json',state)
        state.update(status='complete',finished=time.time())
    except BaseException as error:
        state.update(status='stopped',error=repr(error))
        if child is not None and child.poll() is None:
            p=Path('/proc',str(child.pid));argv=(p/'cmdline').read_bytes().split(b'\0')
            if p.stat().st_uid!=os.geteuid() or str(ENTRY).encode() not in argv or os.getpgid(child.pid)!=child.pid:
                raise RuntimeError('Refuse signal to unverified process')
            os.killpg(child.pid,signal.SIGINT);child.wait(timeout=60)
        raise
    finally:
        if active:shared.stop_private_daemon()
        state['mps_released']=control(args.pipe,'get_server_list').returncode!=0
        write(args.root/'suite_status.json',state);lock.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('suite','worker'),required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path);p.add_argument('--after-suite',type=Path);p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--source-receipt-sha256');args=p.parse_args();args.root=args.root.resolve()
    if args.root.parent!=SIM or not re.fullmatch(r'lgvq_compute_scope_20261029_\d{2}',args.root.name):raise ValueError('New canonical root required')
    if not 0<args.deadline-time.time()<=14400:raise ValueError('Finite four-hour maximum')
    args.pipe='/DATA/DATA1/guest3/.cache/'+args.root.name
    if args.phase=='suite':
        if args.after_suite is None or args.after_suite.resolve()!=SIM/'qwen_fullscope_20261029_02':raise ValueError('Explicit completed new Qwen02 predecessor required')
        args.after_suite=args.after_suite.resolve()
    elif args.output is None or args.output.resolve().parent!=args.root or args.output.name not in [f'lg_repeat{i:02d}' for i in (1,2,3)]:raise ValueError('Exact fresh LG leaf required')
    (suite if args.phase=='suite' else worker)(args)

if __name__=='__main__':main()
