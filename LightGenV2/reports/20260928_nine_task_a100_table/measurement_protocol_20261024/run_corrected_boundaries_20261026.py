"""New approved scope: remove every video's image preparation/H2D, keep other waits.

Qwen remains original B1, no graph/prefetch/parallel input preparation. Tokenizer
CPU intervals remain charged. LG projection is a separately declared estimate,
not optical hardware E2E. Old measurements and raw are never overwritten.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from types import SimpleNamespace
import lgvq_gpu_pipeline_20261026 as base
import qwen_shared_serial_20261026 as shared

ENTRY=Path(__file__).resolve();HERE=ENTRY.parent;REPO=base.REPO
OLD_SOURCES=base.source_map
base.__file__=__file__
base.PIPE='/DATA/DATA1/guest3/.cache/corrected_boundaries_20261026'
N=224

def sources():
    result=OLD_SOURCES();result[ENTRY.relative_to(REPO).as_posix()]=base.sha(ENTRY)
    return result

class ExclusiveGuard(shared.SharedGuard):
    def check(self):
        super().check()
        row=self.audit[-1];row['human_authorized_nonexclusive']=False
        if row['foreign_gpu_processes'] or row['foreign_mps_clients']:raise RuntimeError('Foreign GPU process; preserve own raw only')

def accounted_ms(full_start,last_score,rows):
    removed=0.
    for row in rows:
        assert row['CPU_start']<=row['CPU_end']<=row['H2D_end']
        intervals=row['tokenizer_intervals']
        assert all(row['CPU_start']<=t['start']<=t['end']<=row['CPU_end'] for t in intervals)
        removed+=(row['H2D_end']-row['CPU_start'])*1000-sum((t['end']-t['start'])*1000 for t in intervals)
    value=(last_score-full_start)*1000-removed
    assert value>0
    return value,removed

def qwen_worker(args,guard):
    import numpy as np
    import torch
    import lgvq_live_frontend as live
    from LightGenV2.tasks.t06_video_quality_assessment import quality_token_batch_benchmark_200 as native
    from LightGenV2.tasks.t06_video_quality_assessment.quality_token_common import BoundaryTimer
    pin=json.loads((live.TASK/'runs/simulation/lgvq_mps_e2e_a100_20261024/lab_remaining01/qwen50/qwen_cap50_repeat01/identity.json').read_text())
    command=pin['command'];value=lambda key:command[command.index(key)+1]
    checkpoint=Path(value('--checkpoint'));assert base.sha(checkpoint)==pin['expected_head_sha256']
    opt=SimpleNamespace(model=Path(value('--model-path')),checkpoint=checkpoint,manifest=live.MANIFEST,image_size=448)
    for name,digest in pin['assets'].items():assert base.sha(opt.model/name)==digest
    processor,model,head,scores,all_rows,prompt=native.load_runtime(opt);head.requires_grad_(False);guard.bind()
    assert torch.cuda.get_device_properties(0).multi_processor_count==54
    unique=N if N<=512 else 512;rows=all_rows[:unique];tokenizer=[];processor.tokenizer=base.TokenizerAudit(processor.tokenizer,tokenizer)
    class Timer(BoundaryTimer):
        def __init__(self,m):super().__init__(m);self.records=[]
        def reset(self,expected_calls=1):
            super().reset(expected_calls);e=torch.cuda.Event(enable_timing=True);e.record();self.state.update(start=e,host_start=time.perf_counter())
        def finish(self):
            result=super().finish();self.state['host_end']=time.perf_counter();self.records.append(dict(self.state));return result
    timer=Timer(model);metadata=[];predictions=[];full_start=time.perf_counter()
    try:
        with torch.inference_mode():
            for index in range(N):
                row=rows[index%unique]
                guard.fast();a=time.perf_counter();lo=len(tokenizer)
                cpu,positions=native.prepare_batch([row],processor,prompt,torch.device('cpu'),448)
                b=time.perf_counter();inputs={k:v.to('cuda') for k,v in cpu.items()};torch.cuda.synchronize();c=time.perf_counter()
                _,prediction,_=native.forward_batch(inputs,model,head,scores,timer)
                returned=time.perf_counter();predictions.append(prediction[0]);r=timer.records[-1]
                metadata.append(dict(index=index,source_index=index%unique,cycle=index//unique,sample_id=row['sample_id'],CPU_start=a,CPU_end=b,H2D_end=c,
                    GPU_host_start=r['host_start'],GPU_ready_host=r['host_end'],CPU_result_return=returned,
                    tokenizer_intervals=tokenizer[lo:],frame_positions=positions[0],
                    input_shapes={k:list(v.shape) for k,v in cpu.items()},input_ids=cpu['input_ids'].tolist(),attention_mask=cpu['attention_mask'].tolist()))
                if index:assert a>=metadata[-2]['CPU_result_return']
                del cpu,inputs
                if (index+1)%32==0:base.write(args.output/'status.json',dict(status='Qwen_original_strict_B1',samples=index+1,total=N))
        origin=timer.records[0]['start']
        for row,event in zip(metadata,timer.records):row.update(GPU_start_ms=origin.elapsed_time(event['start']),GPU_end_ms=origin.elapsed_time(event['end_event']),first_TF_ms=origin.elapsed_time(event['vision_event']))
        adjusted,removed=accounted_ms(full_start,metadata[-1]['GPU_ready_host'],metadata)
        model_sum=sum(r['GPU_end_ms']-r['GPU_start_ms'] for r in metadata)
        with_return,removed2=accounted_ms(full_start,metadata[-1]['CPU_result_return'],metadata);assert removed==removed2
        output=np.asarray(predictions,dtype=np.float32);assert output.shape==(N,) and np.isfinite(output).all()
        assert all(np.array_equal(output[:unique],output[k:k+unique]) for k in range(unique,N,unique))
        np.savez_compressed(args.output/'outputs.npz',prediction=output)
        base.write(args.output/'timing.json',dict(status='complete',system='Qwen50',video_calls=N,unique_videos=unique,fields_equivalent=N//16,
            corrected_GPU_ready_ms=adjusted,corrected_CPU_result_ms=with_return,removed_image_preparation_and_H2D_ms=removed,
            full_before_subtraction_ms=(metadata[-1]['GPU_ready_host']-full_start)*1000,GPU_model_window_sum_ms=model_sum,
            tokenizer_CPU_ms=sum((t['end']-t['start'])*1000 for t in tokenizer),
            time16_ms=adjusted*16/N,equivalent_frame_ms=adjusted/(N*4),records=metadata,
            default_original_forward=True,no_graph=True,no_prefetch=True,CPU_preparation_single_video=True,
            every_video_image_preparation_and_original_H2D_excluded=True,other_dispatch_sync_return_waits_retained=True,
            full_native_vision_language_vocabulary_projection_quality_head=True))
        base.write(args.output/'input_files.json',[dict(sample_id=r['sample_id'],video_path=str(r['video_path']),bytes=Path(r['video_path']).stat().st_size,sha256=base.sha(Path(r['video_path']))) for r in rows])
        for name,digest in pin['assets'].items():assert base.sha(opt.model/name)==digest
    finally:timer.close()

def worker(args):
    import torch
    assert os.environ['CUDA_VISIBLE_DEVICES']==base.GPU and os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE']==('50' if args.system=='qwen' else '100')
    assert not args.output.exists();args.output.mkdir();frozen=sources()
    base.write(args.output/'identity.json',dict(system=args.system,repeat=args.repeat,video_calls=N,source_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),sources=frozen,published_commit=os.environ['MEASUREMENT_PUBLISHED_COMMIT'],GPU=base.GPU,cap=50 if args.system=='qwen' else 100))
    base.write(args.output/'status.json',dict(status='loading',pid=os.getpid()))
    guard=ExclusiveGuard(args.deadline,50 if args.system=='qwen' else 100);torch.set_num_threads(8)
    try:
        if args.system=='qwen':qwen_worker(args,guard)
        else:
            import lgvq_physical_projection_20261026 as lg
            lg.worker(args,guard,N)
        assert sources()==frozen,'Frozen source changed'
        guard.check();base.write(args.output/'status.json',dict(status='complete',pid=os.getpid(),finished=time.time()))
    except BaseException as error:
        base.write(args.output/'status.json',dict(status='failed',pid=os.getpid(),error=repr(error)));raise
    finally:guard.close(args.output/'process_audit.json')

def suite(args):
    from first_tf_supplement_20261025 import serialize_mps_commands
    from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
    from run_temporal_remainder import process_snapshot
    serialize_mps_commands();engine.GPU=base.GPU;engine.PIPE=base.PIPE;engine.ROOT=args.output
    assert not args.output.exists() and args.output.is_relative_to(base.ROOT);args.output.mkdir()
    state=dict(status='admission',pid=os.getpid(),deadline=args.deadline,N=N,completed=[],plan=['qwen01','lg01'],single_round_user_requested=True)
    active=False;child=None
    def save():base.write(args.output/'suite_status.json',state)
    def bounded():
        if time.time()>args.deadline-60:raise TimeoutError('Finite corrected-boundary suite')
    save()
    try:
        with (base.ROOT/'.temporal_telemetry_a100.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for name in state['plan']:
                bounded();system='qwen' if name.startswith('qwen') else 'lg';repeat=int(name[-2:])
                for idle in range(12):
                    snap=process_snapshot(base.GPU,-1,base.PIPE)
                    assert not snap['foreign_gpu_processes'] and not snap['foreign_mps_clients']
                    memory=float(subprocess.check_output(['nvidia-smi','-i',base.GPU,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True));assert memory<384
                    time.sleep(.5)
                assert shared.control('get_server_list').returncode!=0
                engine.daemon(True);active=True;leaf=args.output/name
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=base.GPU,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_MPS_PIPE_DIRECTORY=base.PIPE,
                    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE='50' if system=='qwen' else '100',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
                command=['/home/guest3/miniconda3/envs/xml/bin/python','-u',str(ENTRY),'--phase','worker','--system',system,'--repeat',str(repeat),'--output',str(leaf),'--videos',str(N),'--deadline',str(args.deadline)]
                with leaf.with_suffix('.log').open('x') as logfile:
                    child=subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True)
                    state.update(status='running',point=name,child=child.pid);save()
                    while child.poll() is None:
                        bounded();snapshot=process_snapshot(base.GPU,child.pid,base.PIPE)
                        assert not snapshot['foreign_gpu_processes'] and not snapshot['foreign_mps_clients']
                        time.sleep(1)
                shared.stop_private_daemon();active=False
                assert child.returncode==0,'Preserve failed raw before missing-only continuation'
                state['completed'].append(name);save()
            state.update(status='complete',finished=time.time())
    except BaseException as error:
        state.update(status='stopped',error=repr(error))
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGINT);child.wait(timeout=45)
        raise
    finally:
        if active:shared.stop_private_daemon()
        state['mps_released']=True;save()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['suite','worker'],required=True);p.add_argument('--system',choices=['qwen','lg']);p.add_argument('--repeat',type=int);p.add_argument('--output',type=Path,required=True);p.add_argument('--videos',type=int,choices=[224,2048],default=2048);p.add_argument('--deadline',type=float,required=True)
    a=p.parse_args();N=a.videos;a.output=a.output.resolve();(suite if a.phase=='suite' else worker)(a)
