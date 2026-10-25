"""Qwen50 only: prepare ONE video, transfer, infer, return; then next video.

No input preloading/prefetch, no token/embedding reuse, no parallel videos.
Keep both native per-request model windows and the continuous first-model to
last-score timeline, including ALL intervening CPU preparation and waits.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from types import SimpleNamespace
import lgvq_gpu_pipeline_20261026 as base

ORIGINAL_FILE=Path(base.__file__)
original_sources=base.source_map
base.__file__=__file__


def sources():
    result=original_sources();result[ORIGINAL_FILE.relative_to(base.REPO).as_posix()]=base.sha(ORIGINAL_FILE)
    return result


base.source_map=sources


def assert_serial(records):
    for i,r in enumerate(records):
        assert r['index']==i and r['source_index']==i%base.UNIQUE
        assert r['CPU_start']<=r['CPU_end']<=r['H2D_end']<=r['host_start']<r['host_end']<=r['CPU_result_return']
        if i:assert r['CPU_start']>=records[i-1]['CPU_result_return']


def qwen_worker(args,folder,guard):
    import torch
    import numpy as np
    import lgvq_live_frontend as live
    from LightGenV2.tasks.t06_video_quality_assessment import quality_token_batch_benchmark_200 as native
    from LightGenV2.tasks.t06_video_quality_assessment.quality_token_common import BoundaryTimer
    reference=live.TASK/'runs/simulation/lgvq_mps_e2e_a100_20261024/lab_remaining01/qwen50/qwen_cap50_repeat01/identity.json'
    identity=json.loads(reference.read_text());cmd=identity['command'];value=lambda flag:cmd[cmd.index(flag)+1]
    checkpoint=Path(value('--checkpoint'));assert base.sha(checkpoint)==identity['expected_head_sha256']
    opt=SimpleNamespace(model=Path(value('--model-path')),checkpoint=checkpoint,manifest=live.MANIFEST,image_size=448)
    for name,digest in identity['assets'].items():assert base.sha(opt.model/name)==digest
    processor,model,head,scores,rows,prompt=native.load_runtime(opt);head.requires_grad_(False);guard.bind()
    assert torch.cuda.get_device_properties(0).multi_processor_count==54

    class Timer(BoundaryTimer):
        def __init__(self,model):super().__init__(model);self.records=[]
        def reset(self,expected_calls=1):
            super().reset(expected_calls)
            e=torch.cuda.Event(enable_timing=True);e.record();self.state.update(start=e,host_start=time.perf_counter())
        def finish(self):
            result=super().finish();self.state.update(host_end=time.perf_counter())
            self.records.append(dict(self.state));return result

    timer=Timer(model);metadata=[];values=[];tokens=[];processor.tokenizer=base.TokenizerAudit(processor.tokenizer,tokens)
    try:
        with torch.inference_mode():
            for i,index,cycle in base.plan():
                guard.fast();a=time.perf_counter()
                # Single-video CPU preparation occurs HERE, never a prior pass.
                cpu,positions=native.prepare_batch([rows[index]],processor,prompt,torch.device('cpu'),448)
                assert cpu['input_ids'].shape[0]==1
                b=time.perf_counter();inputs={k:v.to('cuda') for k,v in cpu.items()};torch.cuda.synchronize();c=time.perf_counter()
                _,prediction,_=native.forward_batch(inputs,model,head,scores,timer)
                returned=time.perf_counter();values.append(prediction[0])
                metadata.append(dict(index=i,source_index=index,cycle=cycle,sample_id=rows[index]['sample_id'],
                    CPU_start=a,CPU_end=b,H2D_end=c,CPU_result_return=returned,frame_positions=positions[0]))
                del inputs,cpu
                if (i+1)%128==0:base.write(folder/'status.json',dict(status='strict_single_video_CPU_GPU_serial',completed_video_calls=i+1,total=base.TOTAL))
        origin=timer.records[0]['start'];records=[]
        for meta,r in zip(metadata,timer.records):
            records.append(dict(meta,start_ms=origin.elapsed_time(r['start']),first_TF_ms=origin.elapsed_time(r['vision_event']),
                end_ms=origin.elapsed_time(r['end_event']),host_start=r['host_start'],host_end=r['host_end'],
                first_TF_host_start=r['vision_host_started']))
        assert_serial(records)
        for a,b in zip(records,records[1:]):assert a['end_ms']<=b['start_ms']
        array=np.asarray(values,dtype=np.float32).reshape(base.CYCLES,base.UNIQUE)
        assert np.isfinite(array).all() and all(np.array_equal(array[0],r) for r in array[1:])
        np.savez_compressed(folder/'qwen_outputs.npz',prediction=array)
        elapsed=records[-1]['end_ms'];work=sum(r['end_ms']-r['start_ms'] for r in records)
        gaps=sum(b['start_ms']-a['end_ms'] for a,b in zip(records,records[1:]))
        base.write(folder/'qwen_timing.json',dict(status='complete',execution='strict_single_video_CPU_then_H2D_then_GPU_then_CPU_return',
            video_calls=base.TOTAL,unique_videos=base.UNIQUE,cap=50,batch_videos=1,no_prefetch=True,no_prepared_input_reuse=True,
            continuous_GPU_timeline_ms=elapsed,continuous_host_ms=(records[-1]['host_end']-records[0]['host_start'])*1000,
            GPU_model_window_sum_ms=work,all_inter_video_gaps_ms=gaps,
            file_input_to_last_CPU_result_ms=(records[-1]['CPU_result_return']-records[0]['CPU_start'])*1000,
            per_request_first_TF_to_GPU_score_sum_ms=sum(r['end_ms']-r['first_TF_ms'] for r in records),
            time16_ms=elapsed/128,all_inter_video_waits_included=True,terminal_CPU_copy_excluded=True,no_dummy_warmup=True,
            no_Qwen_steady_state_selection=True,records=records,
            prefixes=[dict(video_calls=n,time16_ms=records[n-1]['end_ms']*16/n) for n in base.prefixes()],
            original_forward_function_reused=True,original_BoundaryTimer_reused=True))
        base.write(folder/'external_preparation.json',dict(rows=metadata,tokenizer_calls=tokens,
            excluded_from_GPU_window=False,first_input_CPU_prepare_before_first_GPU_start=True,
            all_subsequent_CPU_prepare_and_H2D_in_continuous_primary=True,
            raw_inputs_GPU_resident=False,neural_features_cached=False,no_multi_video_CPU_preparation=True))
        for name,digest in identity['assets'].items():assert base.sha(opt.model/name)==digest
        return [dict(sample_id=r['sample_id'],video_path=str(r['video_path'])) for r in rows[:base.UNIQUE]]
    finally:timer.close()


base.qwen_worker=qwen_worker
original_worker=base.worker


def worker(args):
    original_worker(args)
    # Source/guard evidence remains original. Append explicit new execution
    # profile only after successful finalization; no active old raw is changed.
    p=args.output/'identity.json';identity=json.loads(p.read_text())
    identity.update(profile='Qwen50_strict_single_video_interleaved_CPU_GPU_512x4_20261026_v1',
        external_preparation_recorded_not_charged=False,all_internal_gaps_retained=True,
        tokenizer_is_CPU_separate_not_added_to_GPU_timeline=False,
        no_prefetch=True,no_prepared_input_reuse=True,CPU_preparation_one_video_at_a_time=True)
    base.write(p,identity)


def suite(args):
    import fcntl
    from first_tf_supplement_20261025 import serialize_mps_commands
    from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
    from LightGenV2.tasks.t06_video_quality_assessment.measure_mps_dataset_once import validate_idle_memory
    import run_temporal_remainder as loading
    serialize_mps_commands();engine.GPU=base.GPU;engine.PIPE=base.PIPE;engine.ROOT=args.output
    if args.output.exists() or not args.output.is_relative_to(base.ROOT) or not 0<args.deadline-time.time()<=10800:
        raise ValueError('New bounded approved task root required')
    args.output.mkdir();state=dict(status='waiting_retained_LG',pid=os.getpid(),deadline=args.deadline,completed=[]);active=False;child=None
    def save():base.write(args.output/'suite_status.json',state)
    def bounded():
        if time.time()>args.deadline-60:raise TimeoutError('Finite strict-serial queue')
    save()
    try:
        with (base.ROOT/'.temporal_telemetry_a100.lock').open('a') as lock:
            while True:
                bounded()
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:time.sleep(5)
            retained=json.loads((args.retained_lg/'suite_status.json').read_text())
            assert retained['status']=='complete' and not Path('/proc',str(retained['pid'])).exists()
            assert retained['completed']==['lg_repeat01','lg_repeat02','lg_repeat03'] and retained['mps_released']
            previous=json.loads((args.previous/'suite_status.json').read_text())
            assert previous['status']=='stopped' and previous['completed']==[]
            assert not Path('/proc',str(previous['pid'])).exists() and not Path('/proc',str(previous['child'])).exists()
            state.update(retained_LG_root=str(args.retained_lg),stopped_Qwen_predecessor=str(args.previous));save()
            for rep in (1,2,3):
                idle=0
                while idle<12:
                    bounded();foreign=engine.base.guard_records(base.GPU)
                    if foreign:idle=0;state.update(status='waiting_foreign_process',repeat=rep,idle_checks=0,foreign=foreign)
                    else:
                        validate_idle_memory(base.GPU);idle+=1;state.update(status='idle_guard',repeat=rep,idle_checks=idle,foreign=[])
                    save();time.sleep(10)
                control=subprocess.run(['nvidia-cuda-mps-control'],input='get_server_list\n',text=True,capture_output=True,env=dict(os.environ,CUDA_MPS_PIPE_DIRECTORY=base.PIPE),timeout=5)
                assert control.returncode!=0,'Private MPS must be released before fresh run'
                engine.daemon(True);active=True;leaf=args.output/f'qwen_repeat{rep:02d}'
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=base.GPU,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_MPS_PIPE_DIRECTORY=base.PIPE,
                    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE='50',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',
                    MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
                cmd=['/home/guest3/miniconda3/envs/xml/bin/python','-u',str(Path(__file__)),'--phase','worker','--system','qwen','--repeat',str(rep),'--output',str(leaf),'--deadline',str(args.deadline)]
                observations=[]
                with leaf.with_suffix('.log').open('x') as log:
                    child=subprocess.Popen(cmd,cwd=base.REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    state.update(status='running',repeat=rep,child=child.pid);save()
                    try:
                        while child.poll() is None:
                            bounded();snap=loading.process_snapshot(base.GPU,child.pid,base.PIPE);observations.append(snap)
                            if snap['foreign_gpu_processes'] or snap['foreign_mps_clients']:raise RuntimeError('Foreign during strict serial Qwen')
                            time.sleep(2)
                    except BaseException:
                        if child.poll() is None:os.killpg(child.pid,signal.SIGINT);child.wait(timeout=45)
                        raise
                base.write(args.output/f'qwen_repeat{rep:02d}_loading_and_process_guard.json',observations)
                engine.daemon(False);active=False
                assert child.returncode==0,'Point failed: preserve raw, no blind restart'
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['suite','worker'],required=True)
    p.add_argument('--system',default='qwen');p.add_argument('--repeat',type=int);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--deadline',type=float,required=True);p.add_argument('--previous',type=Path);p.add_argument('--retained-lg',type=Path)
    a=p.parse_args();a.output=a.output.resolve();(suite if a.phase=='suite' else worker)(a)
