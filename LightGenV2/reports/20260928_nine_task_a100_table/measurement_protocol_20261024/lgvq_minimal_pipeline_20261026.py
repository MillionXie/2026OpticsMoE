"""LG-only minimal-boundary counterpart; preserve dense diagnostic measurements.

Same models, input preparation and complete forward. This removes per-module
timing instrumentation from primary execution, not any model computation. Waits
for the prior suite to truly exit; never reruns its qualified Qwen points.
"""
import argparse
from collections import deque
from contextlib import nullcontext
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time

import lgvq_gpu_pipeline_20261026 as base
ORIGINAL_FILE=Path(base.__file__)
original_source_map=base.source_map
base.__file__=__file__

def sources():
    out=original_source_map();out[ORIGINAL_FILE.relative_to(base.REPO).as_posix()]=base.sha(ORIGINAL_FILE)
    return out
base.source_map=sources

class NoInnerTimer:
    def scope(self,*a,**kw):return nullcontext()

def timed(model,front,position,raw,guard,mode,folder):
    import torch,numpy as np
    from LightGenV2.tasks.t06_video_quality_assessment import lab_runtime as native
    fs,bs,ts=[torch.cuda.Stream() for _ in range(3)]
    origin=torch.cuda.Event(enable_timing=True);origin.record();origin.synchronize()
    q=queue.Queue(maxsize=2);cancel=threading.Event();errors=[];events=[];outputs=[];context={};handles=[]
    for name in ('parallel_router','parallel_optics','serial_router','serial_optics'):
        def count(*args):context['passes']=context.get('passes',0)+1
        handles.append(getattr(model,name).propagation.register_forward_pre_hook(count))
    def first_optics(*args):
        if 'first' not in context:
            e=torch.cuda.Event(enable_timing=True);e.record();context['first']=e
    handles.append(model.parallel_optics.width_to_field.register_forward_pre_hook(first_optics))
    meter=NoInnerTimer();host_start=time.perf_counter()
    def prepare(g):
        with torch.cuda.stream(fs):
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
            # Text stream waits only for this field's begin marker, then runs
            # independently of visual embedding on fs.
            ts.wait_event(a)
            batch=base.native_lg_front(front,position,raw[g%32],meter,ts);b.record()
        return g,batch,a,b
    def producer():
        try:
            with torch.inference_mode():
                for g in range(base.TOTAL//16):
                    if cancel.is_set():return
                    guard.fast();item=prepare(g)
                    while not cancel.is_set():
                        try:q.put(item,timeout=.25);break
                        except queue.Full:guard.fast()
        except BaseException as e:errors.append(e);cancel.set()
    thread=None;pending=deque()
    try:
        if mode=='pipeline':thread=threading.Thread(target=producer,daemon=True);thread.start()
        with torch.inference_mode():
            for g in range(base.TOTAL//16):
                guard.fast()
                if mode=='serial':item=prepare(g)
                else:
                    while True:
                        if errors:raise errors[0]
                        try:item=q.get(timeout=.25);break
                        except queue.Empty:guard.fast()
                index,batch,a,b=item;assert index==g
                with torch.cuda.stream(bs):
                    bs.wait_event(b);context.clear();c=torch.cuda.Event(enable_timing=True);d=torch.cuda.Event(enable_timing=True);c.record()
                    pred=native.forward(model,batch)['prediction'];d.record()
                    if context.get('passes')!=6 or 'first' not in context:raise RuntimeError('Actual original six passes or first optical boundary missing')
                    events.append(dict(field=g,front_start=a,front_end=b,body_start=c,first_optic=context['first'],body_end=d,optical_passes=context['passes']))
                outputs.append(pred);pending.append((d,batch))
                if mode=='serial':d.synchronize()
                elif len(pending)>2:pending.popleft()[0].synchronize()
                if (g+1)%16==0:base.write(folder/'status.json',dict(status='minimal_boundary_GPU_timing',mode=mode,completed_video_calls=(g+1)*16,total=base.TOTAL))
            events[-1]['body_end'].synchronize();host_end=time.perf_counter()
        if thread:thread.join(10)
        if errors:raise errors[0]
        torch.cuda.synchronize()
        records=[dict(field=e['field'],optical_passes=e['optical_passes'],**{k:origin.elapsed_time(e[k]) for k in ['front_start','front_end','body_start','first_optic','body_end']}) for e in events]
        begin=records[0]['front_start'];end=records[-1]['body_end'];elapsed=end-begin
        ov=sum(base.overlap([(r['front_start'],r['front_end']) for r in records[i+1:]],a['body_start'],a['body_end']) for i,a in enumerate(records))
        array=torch.cat(outputs).float().cpu().numpy();assert array.shape==(128,16) and np.isfinite(array).all()
        if not all(np.array_equal(array[:32],array[32*c:32*(c+1)]) for c in (1,2,3)):raise RuntimeError('Repeated full-field output changed')
        np.savez_compressed(folder/(mode+'_outputs.npz'),prediction=array)
        base.write(folder/(mode+'_timing.json'),dict(status='complete',instrumentation='minimal_boundary_events_no_per_module_timing',mode=mode,
            video_calls=2048,unique_videos=512,fields=128,frames_per_field=64,
            first_GPU_front_start_ms=begin,last_GPU_score_ms=end,continuous_GPU_timeline_ms=elapsed,dispatch_to_last_score_host_ms=1000*(host_end-host_start),
            time16_ms=elapsed/128,equivalent_frame_ms=elapsed/8192,actual_later_front_overlapping_body_ms=ov,
            first_group_ms=records[0]['body_end']-begin,
            prefixes=[dict(video_calls=n,time16_ms=(records[n//16-1]['body_end']-begin)*16/n) for n in base.prefixes()],
            steady=base.steady_report([r['body_end']-begin for r in records]),records=records,
            actual_complete_software_includes_FFT=True,physical_optical_E2E_measured=False,no_dummy_warmup=True,
            text_embedding_each_field_on_separate_stream=True,input_adapters_still_in_original_body_stream=True,
            fixed_optical_ms_per_field_reference=6*(1000/1400+.3+.0307)))
        return array
    finally:
        cancel.set()
        if thread:thread.join(10)
        for h in handles:h.remove()

base.lg_timed=timed

def suite(args):
    import fcntl,json
    from first_tf_supplement_20261025 import serialize_mps_commands
    from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
    from LightGenV2.tasks.t06_video_quality_assessment.measure_mps_dataset_once import validate_idle_memory
    import run_temporal_remainder as loading
    serialize_mps_commands();engine.GPU=base.GPU;engine.PIPE=base.PIPE;engine.ROOT=args.output
    if args.output.exists() or not args.output.is_relative_to(base.ROOT):raise ValueError('New task root required')
    args.output.mkdir();state=dict(status='waiting_previous_suite',pid=os.getpid(),completed=[],deadline=args.deadline);active=False;child=None
    def save():base.write(args.output/'suite_status.json',state)
    def bounded():
        if time.time()>args.deadline-60:raise TimeoutError('Finite minimal-boundary run')
    save()
    try:
        with (base.ROOT/'.temporal_telemetry_a100.lock').open('a') as lock:
            while True:
                bounded()
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:time.sleep(5)
            previous=json.loads((args.previous/'suite_status.json').read_text())
            if previous['status']!='complete' or Path('/proc',str(previous['pid'])).exists():raise RuntimeError('Previous suite not complete and truly exited')
            if previous['completed']!=['lg_repeat01','lg_repeat02','lg_repeat03','qwen_repeat01','qwen_repeat02','qwen_repeat03']:raise RuntimeError('Retain all qualified Qwen points')
            state['retained_Qwen_root']=str(args.previous);save()
            for rep in (1,2,3):
                for idle in range(12):
                    bounded()
                    if engine.base.guard_records(base.GPU):raise RuntimeError('Foreign before minimal-boundary run')
                    validate_idle_memory(base.GPU);state.update(status='idle_guard',repeat=rep,idle_checks=idle+1);save();time.sleep(10)
                control=subprocess.run(['nvidia-cuda-mps-control'],input='get_server_list\n',text=True,capture_output=True,env=dict(os.environ,CUDA_MPS_PIPE_DIRECTORY=base.PIPE),timeout=5)
                if control.returncode==0:raise RuntimeError('Private MPS not released')
                engine.daemon(True);active=True;leaf=args.output/f'lg_repeat{rep:02d}'
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=base.GPU,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_MPS_PIPE_DIRECTORY=base.PIPE,CUDA_MPS_ACTIVE_THREAD_PERCENTAGE='100',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
                cmd=['/home/guest3/miniconda3/envs/xml/bin/python','-u',str(Path(__file__)),'--phase','worker','--system','lg','--repeat',str(rep),'--output',str(leaf),'--deadline',str(args.deadline)]
                observations=[]
                with leaf.with_suffix('.log').open('x') as log:
                    child=subprocess.Popen(cmd,cwd=base.REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    state.update(status='running',repeat=rep,child=child.pid);save()
                    try:
                        while child.poll() is None:
                            bounded();snap=loading.process_snapshot(base.GPU,child.pid,base.PIPE);observations.append(snap)
                            if snap['foreign_gpu_processes'] or snap['foreign_mps_clients']:raise RuntimeError('Foreign during minimal-boundary run')
                            time.sleep(2)
                    except BaseException:
                        if child.poll() is None:os.killpg(child.pid,signal.SIGINT);child.wait(timeout=45)
                        raise
                base.write(args.output/f'lg_repeat{rep:02d}_loading_and_process_guard.json',observations)
                engine.daemon(False);active=False
                if child.returncode:raise RuntimeError('Minimal-boundary point failed; no blind restart')
                state['completed'].append(leaf.name);save()
            state.update(status='complete',finished=time.time())
    except BaseException as e:state.update(status='stopped',error=repr(e));raise
    finally:
        if active:
            try:engine.daemon(False);state['mps_released']=True
            except BaseException as e:state['mps_release_error']=repr(e)
        else:state['mps_released']=True
        save()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['suite','worker'],required=True);p.add_argument('--system',default='lg');p.add_argument('--repeat',type=int)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path);p.add_argument('--deadline',type=float,required=True)
    a=p.parse_args();a.output=a.output.resolve();(suite if a.phase=='suite' else base.worker)(a)
