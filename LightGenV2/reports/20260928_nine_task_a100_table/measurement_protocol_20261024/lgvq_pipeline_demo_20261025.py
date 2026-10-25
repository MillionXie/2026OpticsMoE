"""Authorized Temporal-only resident timing demo; never train or run Qwen.

CPU preparation is bounded and input-order preserving. The original decoder,
processor, quality feature function, patch/position weights and complete optical
body are reused. A geometry-only position buffer is built at model setup; no
sample feature or prediction is cached online. All timed inputs are decoded
again. Software E2E and an explicitly segmented optical projection are distinct.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(REPO))
GPU='GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55'
PIPE='/DATA/DATA1/guest3/.cache/nine_task_input_e2e_20261024'
ROOT=REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation'


def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def field_plan(n=40):
    if not 16<=n<=64:raise ValueError('Small real Temporal demo only')
    indices=list(range(n));flags=[True]*n;pad=(-n)%16
    indices+=indices[:pad];flags+=[False]*pad
    return [(indices[i:i+16],flags[i:i+16]) for i in range(0,len(indices),16)]


def setup_cpu(model_path,barrier):
    # Spawned preparation workers cannot initialize CUDA or a full model.
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TOKENIZERS_PARALLELISM']='false'
    import torch,cv2
    torch.set_num_threads(1);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    from transformers import AutoProcessor
    global CPU_PROCESSOR
    CPU_PROCESSOR=AutoProcessor.from_pretrained(model_path,local_files_only=True)
    barrier.wait(timeout=150)


def cpu_ping():return os.getpid()


def prepare_video(row):
    import torch
    from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54 import cache_qwen_front as original
    start=time.perf_counter();images=original.decode_frames(Path(row['video_path']),4);decoded=time.perf_counter()
    inputs=CPU_PROCESSOR(text=['x']*4,images=images,return_tensors='pt');processed=time.perf_counter()
    quality=original.quality_tokens_from_images(images,video_count=1,frame_count=4,output_grid=7)
    end=time.perf_counter()
    if torch.cuda.is_initialized():raise RuntimeError('CPU preparation touched CUDA')
    return dict(sample_id=row['sample_id'],pixels=inputs['pixel_values'],grid=inputs['image_grid_thw'],
        quality=quality,pid=os.getpid(),start=start,decode_end=decoded,processor_end=processed,end=end)


class EventMeter:
    """Observe original neural/FFT scopes without a device drain per scope."""
    def __init__(self):self._restore=[];self.rows=[];self.depth=0;self.field=-1
    def wrap(self,owner,method,name,*,family='neural',charged=True,gpu=True):
        original=getattr(owner,method)
        def observed(*args,**kw):
            with self.scope(name,family,charged,gpu):return original(*args,**kw)
        setattr(owner,method,observed);self._restore.append((owner,method,original))
    @contextmanager
    def scope(self,name,family='neural',charged=True,gpu=True):
        import torch
        if self.depth:
            yield;return
        self.depth+=1;a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
        started=time.perf_counter();a.record()
        try:yield
        finally:
            b.record();self.rows.append(dict(field=self.field,stage=name,family=family,charged=charged,
                enqueue_start=started,enqueue_end=time.perf_counter(),start_event=a,end_event=b));self.depth-=1
    def records(self,begin):
        return [{k:v for k,v in row.items() if k not in ('start_event','end_event')}|
                dict(cuda_event_ms=row['start_event'].elapsed_time(row['end_event'])) for row in self.rows[begin:]]
    def restore(self):
        for owner,method,original in reversed(self._restore):setattr(owner,method,original)


class PreparedFront:
    def __init__(self,front):
        import torch
        self.front=front;self.batch_videos=16
        grid=torch.tensor([[1,28,28]]*4,device=front.device,dtype=torch.long)
        start=time.perf_counter()
        with torch.inference_mode():self.position4=front.visual.fast_pos_embed_interpolate(grid).detach()
        torch.cuda.synchronize();self.setup_position_s=time.perf_counter()-start
        self.geometry_grid=[[1,28,28]]*4
    def build(self,prepared,meter):
        import torch
        from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54 import cache_qwen_front as native
        if len(prepared)!=16 or any(p['grid'].tolist()!=self.geometry_grid for p in prepared):
            raise ValueError('Original four-frame fixed geometry differs')
        visuals=[]
        for offset in range(0,16,self.batch_videos):
            items=prepared[offset:offset+self.batch_videos]
            with meter.scope('frontend.input_pixels_H2D'):
                pixels=torch.cat([p['pixels'] for p in items]).to(self.front.device,
                    dtype=next(self.front.visual.patch_embed.parameters()).dtype)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                with meter.scope('frontend.native_patch'):
                    hidden=self.front.visual.patch_embed(pixels)
                with meter.scope('frontend.geometry_position_add'):
                    hidden=hidden+self.position4.repeat(len(items),1).to(dtype=hidden.dtype)
                with meter.scope('frontend.native_mean_pool'):
                    pooled=native.pool_qwen_front_tokens(hidden,image_count=4*len(items),output_grid=7)
            # Original producer returns FP16 before the body; keep that rounding.
            visuals.append(pooled.reshape(len(items),4,49,1024).half())
        text_start=time.perf_counter()
        rendered=native.render_prompt(self.front.tokenizer,target_name='temporal',prompt=native.TARGET_PROMPTS['temporal'])
        tokens=self.front.tokenizer([rendered],padding=True,return_tensors='pt',add_special_tokens=False)
        tokenizer_end=time.perf_counter()
        with meter.scope('frontend.language_embedding'):
            language=self.front.embedding(tokens['input_ids'].to(self.front.device)).half()
        with meter.scope('frontend.quality_H2D'):
            quality=torch.cat([p['quality'] for p in prepared]).unsqueeze(0).to(self.front.device)
            mask=tokens['attention_mask'].bool().to(self.front.device)
        return dict(vision_tokens=torch.cat(visuals).unsqueeze(0),quality_tokens=quality,
                    language_tokens=language,language_mask=mask),dict(tokenizer_start=text_start,tokenizer_end=tokenizer_end,
                    input_ids=tokens['input_ids'].tolist(),attention_mask=tokens['attention_mask'].tolist())


def get_prepared(pool,rows):return [pool.submit(prepare_video,r) for r in rows]


def pool_context(workers,model_path):
    ctx=mp.get_context('spawn');barrier=ctx.Barrier(workers+1)
    pool=ProcessPoolExecutor(max_workers=workers,mp_context=ctx,initializer=setup_cpu,initargs=(str(model_path),barrier))
    futures=[pool.submit(cpu_ping) for _ in range(workers)]
    try:barrier.wait(timeout=150);[f.result(timeout=10) for f in futures]
    except BaseException:pool.shutdown(wait=False,cancel_futures=True);raise
    return pool


def compare_inputs(original,candidate):
    import torch
    result={}
    for key in original:
        a=original[key].detach().cpu();b=candidate[key].detach().cpu()
        equal=a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
        result[key]=dict(equal=equal,original_dtype=str(a.dtype),candidate_dtype=str(b.dtype),shape=list(a.shape),
                         max_abs=0. if equal else float((a.float()-b.float()).abs().max()))
    return result


def worker(root,deadline):
    if (os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU or os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE')!='100'
            or (root/'status.json').exists()):raise ValueError('Fresh admitted original A100/MPS100 worker only')
    import torch
    import numpy as np
    import lgvq_live_frontend as live
    from first_tf_supplement_20261025 import serialize_mps_commands
    from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
    from LightGenV2.tasks.t06_video_quality_assessment import lab_runtime as original
    from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.data import read_manifest
    serialize_mps_commands();engine.GPU=GPU;engine.PIPE=PIPE
    torch.set_num_threads(8)
    guard=engine.MPSGuard(argparse.Namespace(deadline=deadline));base_check=guard.check
    def check():
        base_check()
        raw=subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=uuid,name,power.limit',
            '--format=csv,noheader,nounits'],text=True,timeout=5)
        parts=raw.strip().split(',')
        if parts[0].strip()!=GPU or parts[1].strip()!='NVIDIA A100-PCIE-40GB' or abs(float(parts[2])-250)>.1:
            raise RuntimeError('Physical platform changed')
        guard.audit[-1]['platform']=dict(uuid=GPU,name=parts[1].strip(),power_limit_w=float(parts[2]))
    guard.check=check;guard.check();pool=None;meter=None
    def status(phase,**more):write(root/'status.json',dict(status=phase,pid=os.getpid(),unix=time.time(),**more))
    status('loading');all_runs=[];checks=[]
    try:
        if live.sha(live.MANIFEST)!=live.MANIFEST_SHA:raise ValueError('Original TEST identity differs')
        tests=[r for r in read_manifest(live.MANIFEST) if r.split=='test'][:40]
        rows=[dict(sample_id=r.sample_id,video_path=str(r.video_path)) for r in tests]
        before={r['sample_id']:(Path(r['video_path']).stat().st_size,Path(r['video_path']).stat().st_mtime_ns) for r in rows}
        reference_path=live.TASK/'runs/simulation/lgvq_mps_e2e_a100_20261024/qwen100_repaired01/qwen_cap100_repeat01/report.json'
        input_reference=json.loads(reference_path.read_text());reference_sha=sha(reference_path)
        plan=field_plan(40)
        front=live.LiveFront('cuda:0');checkpoint=live.TASK/'runs/simulation/multivideo16x4_rank_s163/best_checkpoint.pt'
        model,settings=original.load_model('temporal',checkpoint,'cuda:0');model.requires_grad_(False)
        if torch.cuda.get_device_properties(0).multi_processor_count!=108:raise RuntimeError('Wrong MPS100 allocation')
        guard.bind();write(root/'admission_bound.json',dict(pid=os.getpid(),unix=time.time()))
        prepared_front=PreparedFront(front)
        bound={**live.sources(),Path(__file__).relative_to(REPO).as_posix():sha(__file__)}
        identity=dict(command=sys.argv,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
            sources=bound,manifest_sha256=live.MANIFEST_SHA,body_sha256=live.PINS['temporal'][1],gpu_uuid=GPU,mps_percent=100,
            profile='LGVQ_Temporal_parallel_pipeline_demo_v1',videos=40,fields=3,executed_slots=48,frame_count=4,
            no_input_feature_cache=True,position_cache='fixed model geometry only, never sample features',
            position_setup_seconds=prepared_front.setup_position_s,original_geometry=prepared_front.geometry_grid,
            input_reference=str(reference_path),input_reference_sha256=reference_sha,
            loaded_model_resident_demo=True,validation_not_part_of_timing=True,accuracy_not_evaluated=True,
            optical_fixed_ms_per_field=6*1.0447,actual_software_E2E_is_not_physical_optical_E2E=True)
        write(root/'identity.json',identity)
        pool=pool_context(8,front.model_path)
        status('validating_two_real_fields')
        with torch.inference_mode():
            reference=[]
            for index in (0,2):
                guard.fast();ids,valid=plan[index];raw,meta=front.build([tests[i] for i in ids],'temporal')
                expected=original.forward(model,raw)['prediction'].detach().cpu()
                prepared=[f.result(timeout=90) for f in get_prepared(pool,[rows[i] for i in ids])]
                reference.append((index,raw,expected,prepared))
            accepted=False
            for batch_videos in (16,1):
                prepared_front.batch_videos=batch_videos;attempt=[]
                for index,raw,expected,prepared in reference:
                    diagnostic=EventMeter();candidate,_=prepared_front.build(prepared,diagnostic)
                    actual=original.forward(model,candidate)['prediction'].detach().cpu()
                    equal=compare_inputs(raw,candidate);same=torch.equal(expected,actual)
                    attempt.append(dict(field=index,valid=plan[index][1],input_checks=equal,
                        full_field_output_equal=same,max_output_abs=float((expected-actual).abs().max())))
                passed=all(c['full_field_output_equal'] and all(v['equal'] for v in c['input_checks'].values()) for c in attempt)
                checks.append(dict(batch_videos=batch_videos,passed=passed,checks=attempt));write(root/'consistency.json',checks)
                if passed:accepted=True;break
            if not accepted:raise RuntimeError('Optimized original inputs/output differ; no tolerance relaxation')
            del reference
            status('timing_demo',accepted_GPU_front_batch=prepared_front.batch_videos)
            for workers in (8,16):
                if workers==16:
                    pool.shutdown(wait=True);pool=pool_context(16,front.model_path)
                for pipeline in ((False,True) if workers==8 else (True,)):
                    for repeat in (1,2):
                        guard.fast();label='workers%d_%s_repeat%d'%(workers,'pipeline' if pipeline else 'sequential_fields',repeat)
                        meter=EventMeter();live.instrument_body(model,meter)
                        pending={};records=[];outputs=[];software_start=time.perf_counter()
                        for index in range(min(2 if pipeline else 1,len(plan))):
                            pending[index]=get_prepared(pool,[rows[i] for i in plan[index][0]])
                        for index,(indices,valid) in enumerate(plan):
                            guard.fast();prepared=[f.result(timeout=90) for f in pending.pop(index)]
                            # Submit at most the next field while the current field runs.
                            nxt=index+(2 if pipeline else 1)
                            if pipeline and nxt<len(plan):pending[nxt]=get_prepared(pool,[rows[i] for i in plan[nxt][0]])
                            ready=time.perf_counter();meter.field=index;begin=len(meter.rows)
                            batch,meta=prepared_front.build(prepared,meter);front_enqueued=time.perf_counter()
                            result=original.forward(model,batch);torch.cuda.synchronize();gpu_done=time.perf_counter()
                            host=result['prediction'].detach().float().cpu();returned=time.perf_counter()
                            stages=meter.records(begin)
                            if host.shape!=(1,16) or not torch.isfinite(host).all():raise RuntimeError('Invalid complete-field output')
                            if sum(r['family']=='optical_simulation_FFT' for r in stages)!=6:raise RuntimeError('Six full-field propagations missing')
                            outputs.append(host.numpy());records.append(dict(field=index,indices=indices,valid=valid,
                                sample_ids=[rows[i]['sample_id'] for i in indices],cpu_ready=ready,front_enqueued=front_enqueued,
                                gpu_done=gpu_done,cpu_result_return=returned,metadata=meta,
                                CPU_preparation=[{k:v for k,v in p.items() if k not in ('pixels','grid','quality')} for p in prepared],stages=stages))
                            if not pipeline and nxt<len(plan):pending[nxt]=get_prepared(pool,[rows[i] for i in plan[nxt][0]])
                        software_end=records[-1]['cpu_result_return'];meter.restore();meter=None
                        arr=np.concatenate(outputs);np.savez_compressed(root/(label+'.npz'),prediction=arr,valid=np.asarray([v for _,flags in plan for v in flags]))
                        duration=software_end-software_start
                        observed_front=sum(sum(s['cuda_event_ms'] for s in r['stages'] if s['stage'].startswith('frontend.')) for r in records)
                        selected_body=sum(sum(s['cuda_event_ms'] for s in r['stages'] if not s['stage'].startswith('frontend.') and s['charged']) for r in records)
                        # Stage-sum reference only. Never rename it pipelined E2E.
                        stage_projection=observed_front+selected_body+len(plan)*6*1.0447
                        run=dict(label=label,workers=workers,pipeline=pipeline,repeat=repeat,videos=40,fields=3,executed_slots=48,
                            actual_software_E2E_start=software_start,actual_software_E2E_end=software_end,
                            actual_software_E2E_seconds=duration,actual_software_E2E16_ms=duration*1000*16/40,
                            actual_software_equivalent_frame_ms=duration*1000/160,
                            post_first_field_cadence16_ms=(records[-1]['cpu_result_return']-records[0]['cpu_result_return'])*1000/2,
                            post_first_field_is_steady_cadence_not_request_latency=True,
                            recorded_GPU_front_total_ms=observed_front,selected_body_total_ms=selected_body,
                            serial_selected_stage_plus_fixed_optical16_ms=stage_projection*16/40,
                            projection_excludes_CPU_and_is_NOT_E2E=True,records=records)
                        write(root/(label+'.json'),run);all_runs.append(run)
                        if not np.array_equal(arr,all_outputs if len(all_runs)>1 else arr):raise RuntimeError('Timed repeats/configuration outputs differ')
                        all_outputs=arr.copy();status('timing_demo',completed=[r['label'] for r in all_runs])
            if any(live.sha(REPO/p)!=digest for p,digest in bound.items()):raise RuntimeError('Frozen source changed')
        for row in rows:
            stat=Path(row['video_path']).stat();row['sha256']=sha(row['video_path'])
            if (stat.st_size,stat.st_mtime_ns)!=before[row['sample_id']] or row['sha256']!=input_reference['input_files'][row['sample_id']]['sha256']:
                raise RuntimeError('Original input bytes changed or differ from frozen Qwen input')
        if sha(reference_path)!=reference_sha:raise RuntimeError('Qualified input reference changed')
        write(root/'input_files.json',rows)
        summary=dict(status='complete',demo_only=True,full_TEST_not_repeated=True,Qwen_not_executed=True,
            consistency=checks,outputs_bitwise_equal_across_demo_runs=True,runs=[{k:v for k,v in r.items() if k!='records'} for r in all_runs],
            Qwen50_frozen_E2E16_ms=6156.869285306893,
            ratios_to_frozen_Qwen50=[dict(label=r['label'],ratio=6156.869285306893/r['actual_software_E2E16_ms']) for r in all_runs],
            original_formal_costs_not_overwritten=True,physical_optical_E2E_not_measured=True)
        write(root/'summary.json',summary);status('complete',runs=len(all_runs))
    except BaseException as error:status('failed',error=repr(error),completed=[r['label'] for r in all_runs]);raise
    finally:
        if meter is not None:meter.restore()
        if pool is not None:pool.shutdown(wait=True,cancel_futures=True)
        guard.close(root/'process_audit.json')


def suite(root,deadline):
    import fcntl
    from first_tf_supplement_20261025 import serialize_mps_commands
    from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
    from LightGenV2.tasks.t06_video_quality_assessment.measure_mps_dataset_once import validate_idle_memory
    import run_temporal_remainder as loading
    serialize_mps_commands();engine.GPU=GPU;engine.PIPE=PIPE;engine.ROOT=root
    if root.exists() or not root.is_relative_to(ROOT) or not 0<deadline-time.time()<=1800:raise ValueError('New bounded demo root required')
    root.mkdir();state=dict(status='admission',pid=os.getpid(),deadline=deadline,command=sys.argv);active=False;child=None
    def save():write(root/'suite_status.json',state)
    def bounded():
        if time.time()>deadline-60:raise TimeoutError('Bounded Temporal-only demo budget')
    save()
    try:
        with (ROOT/'.temporal_telemetry_a100.lock').open('a') as lock:
            while True:
                bounded()
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:time.sleep(5)
            idle=0
            while idle<12:
                bounded();foreign=engine.base.guard_records(GPU)
                if foreign:raise RuntimeError('A100 occupied; do not load or start CPU preparation')
                validate_idle_memory(GPU);idle+=1;state.update(status='idle_guard',idle_checks=idle);save();time.sleep(10)
            control=subprocess.run(['nvidia-cuda-mps-control'],input='get_server_list\n',text=True,capture_output=True,timeout=5,
                env=dict(os.environ,CUDA_MPS_PIPE_DIRECTORY=PIPE))
            if control.returncode==0:raise RuntimeError('Prior private MPS must be released')
            engine.daemon(True);active=True
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_MPS_PIPE_DIRECTORY=PIPE,
                CUDA_MPS_ACTIVE_THREAD_PERCENTAGE='100',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
            cmd=['/home/guest3/miniconda3/envs/xml/bin/python','-u',str(Path(__file__)), '--worker','--output',str(root),'--deadline',str(deadline)]
            with (root/'worker.log').open('x') as log:
                child=subprocess.Popen(cmd,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state.update(status='running',worker_pid=child.pid);save();observations=[]
                try:
                    while child.poll() is None:
                        bounded()
                        if not (root/'admission_bound.json').exists():
                            snapshot=loading.process_snapshot(GPU,child.pid,PIPE);observations.append(snapshot)
                            if snapshot['foreign_gpu_processes'] or snapshot['foreign_mps_clients']:raise RuntimeError('Foreign A100 during loading')
                        time.sleep(2)
                except BaseException:
                    if child.poll() is None:os.killpg(child.pid,signal.SIGINT);child.wait(timeout=30)
                    raise
                finally:write(root/'loading_process_audit.json',observations)
            if child.returncode:raise RuntimeError('Demo worker failed; preserve raw, no blind restart')
            state.update(status='complete',finished=time.time())
    except BaseException as error:state.update(status='stopped',error=repr(error));raise
    finally:
        if active:
            try:engine.daemon(False);state['mps_released']=True
            except BaseException as error:state['mps_release_error']=repr(error)
        save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',action='store_true')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--deadline',type=float,required=True);a=p.parse_args()
    (worker if a.worker else suite)(a.output.resolve(),a.deadline)
