"""GPU-input-resident LG Temporal dual-stream vs native serial Qwen50.

This is a NEW repeated-input throughput protocol, not file-input E2E or optical
hardware acquisition. CPU decode/processor/tokenizer/quality and input transfer
are executed and separately recorded. Every measured forward recomputes native
embedding/network/output; only raw model inputs and constant geometry reside.
"""
from __future__ import annotations
import argparse
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(REPO))
from lgvq_pipeline_demo_20261025 import GPU,PIPE,ROOT,write,sha
UNIQUE=512;CYCLES=4;TOTAL=UNIQUE*CYCLES


def plan():
    return [(i,i%UNIQUE,i//UNIQUE) for i in range(TOTAL)]


def prefixes():return (256,512,1024,2048)


def overlap(intervals,a,b):
    merged=[]
    for x,y in sorted((max(a,x),min(b,y)) for x,y in intervals if min(b,y)>max(a,x)):
        if merged and x<=merged[-1][1]:merged[-1][1]=max(y,merged[-1][1])
        else:merged.append([x,y])
    return sum(y-x for x,y in merged)


def steady_report(group_end_ms):
    # Fixed full run; never truncate at a favorable stable/fast window.
    durations=[group_end_ms[0]]+[b-a for a,b in zip(group_end_ms,group_end_ms[1:])]
    windows=[]
    for start in range(0,len(durations)-7,8):
        values=durations[start:start+8];mean=sum(values)/8
        cv=(sum((v-mean)**2 for v in values)/8)**.5/mean
        windows.append(dict(first_group=start,last_group=start+7,mean_ms=mean,cv=cv))
    stable=None
    for i in range(2,len(windows)):
        three=windows[i-2:i+1];means=[x['mean_ms'] for x in three]
        if max(x['cv'] for x in three)<=.05 and max(means)/min(means)-1<=.03:
            stable=(three[-1]['last_group']+1)*16;break
    return dict(criteria='three consecutive nonoverlapping 8-group windows, each CV<=5%, max/min means-1<=3%',
        first_qualified_prefix_videos=stable,all_windows=windows,
        warning='finite repeated-input throughput stability, not new independent samples or proof of hardware optical overlap')


def source_map():
    import lgvq_live_frontend as live
    paths=[Path(__file__),HERE/'lgvq_pipeline_demo_20261025.py',HERE/'first_tf_supplement_20261025.py',
        live.TASK/'quality_token_batch_benchmark_200.py',live.TASK/'quality_token_common.py',
        live.TASK/'evaluate_temporal_baseline_batches.py']
    return {**live.sources(),**{p.relative_to(REPO).as_posix():sha(p) for p in paths}}


class TokenizerAudit:
    def __init__(self,original,records):self.original=original;self.records=records
    def __getattr__(self,name):return getattr(self.original,name)
    def __call__(self,*a,**kw):
        start=time.perf_counter();out=self.original(*a,**kw)
        self.records.append(dict(start=start,end=time.perf_counter()));return out


def model_guard(deadline,cap):
    import torch
    from first_tf_supplement_20261025 import serialize_mps_commands
    from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
    serialize_mps_commands();engine.GPU=GPU;engine.PIPE=PIPE
    guard=engine.MPSGuard(SimpleNamespace(deadline=deadline));original=guard.check
    def check():
        original()
        value=subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=uuid,name,power.limit','--format=csv,noheader,nounits'],text=True,timeout=5)
        u,n,w=[x.strip() for x in value.strip().split(',')]
        if u!=GPU or n!='NVIDIA A100-PCIE-40GB' or abs(float(w)-250)>.1:raise RuntimeError('Original physical A100250W required')
        guard.audit[-1]['platform']=dict(uuid=u,name=n,power_limit_w=float(w),requested_cap=cap)
    guard.check=check;guard.check()
    return guard


def prep_lg(front,tests,folder,guard):
    import torch
    import lgvq_live_frontend as live
    from lgvq_pipeline_demo_20261025 import pool_context,get_prepared
    from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54 import cache_qwen_front as native
    rows=[dict(sample_id=r.sample_id,video_path=str(r.video_path)) for r in tests[:UNIQUE]]
    pool=pool_context(16,front.model_path);records=[];ready=[];tokenizer=[]
    try:
        # Bounded preparation queue, no per-sample neural feature cache.
        pending={i:get_prepared(pool,rows[i*16:(i+1)*16]) for i in range(2)}
        for g in range(UNIQUE//16):
            guard.fast();prepared=[f.result(timeout=150) for f in pending.pop(g)]
            if g+2<UNIQUE//16:pending[g+2]=get_prepared(pool,rows[(g+2)*16:(g+3)*16])
            if any(p['grid'].tolist()!=[[1,28,28]]*4 for p in prepared):raise RuntimeError('Original geometry changed')
            start=time.perf_counter();pixels=torch.cat([p['pixels'] for p in prepared]).to('cuda',dtype=torch.bfloat16)
            quality=torch.cat([p['quality'] for p in prepared]).unsqueeze(0).to('cuda')
            a=time.perf_counter();rendered=native.render_prompt(front.tokenizer,target_name='temporal',prompt=native.TARGET_PROMPTS['temporal'])
            tok=front.tokenizer([rendered],padding=True,return_tensors='pt',add_special_tokens=False);b=time.perf_counter()
            tokenizer.append(dict(group=g,start=a,end=b))
            ids=tok['input_ids'].to('cuda');mask=tok['attention_mask'].bool().to('cuda');torch.cuda.synchronize()
            ready.append(dict(pixels=pixels,quality=quality,ids=ids,mask=mask))
            records.append(dict(group=g,CPU_preparation=[{k:v for k,v in p.items() if k not in ('pixels','grid','quality')} for p in prepared],
                packing_transfer_and_tokenizer_start=start,end=time.perf_counter(),input_ids=tok['input_ids'].tolist(),mask=tok['attention_mask'].tolist()))
            write(folder/'status.json',dict(status='preparing_original_LG_inputs',completed_unique=(g+1)*16,total_unique=UNIQUE))
    finally:pool.shutdown(wait=True,cancel_futures=True)
    write(folder/'external_preparation.json',dict(rows=records,tokenizer=tokenizer,excluded_from_GPU_window=True,
        raw_inputs_GPU_resident=True,neural_features_cached=False))
    return ready,rows


def native_lg_front(front,position,raw,meter,text_stream):
    import torch
    from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54 import cache_qwen_front as native
    # Text has no dependency on vision. Compute each group's embedding, not a
    # hoisted embedding cache, on a separate stream inside the timed interval.
    with torch.cuda.stream(text_stream):
        with meter.scope('frontend.text_embedding'):language=front.embedding(raw['ids']).half()
        text_done=torch.cuda.Event();text_done.record()
    with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
        with meter.scope('frontend.conv_embedding'):hidden=front.visual.patch_embed(raw['pixels'])
        with meter.scope('frontend.image_position'):hidden=hidden+position.to(dtype=hidden.dtype)
        with meter.scope('frontend.mean_pool'):pooled=native.pool_qwen_front_tokens(hidden,image_count=64,output_grid=7)
    torch.cuda.current_stream().wait_event(text_done)
    return dict(vision_tokens=pooled.reshape(1,16,4,49,1024).half(),quality_tokens=raw['quality'],language_tokens=language,language_mask=raw['mask'])


def lg_timed(model,front,position,raw,guard,mode,folder):
    import torch,numpy as np
    import lgvq_live_frontend as live
    from lgvq_pipeline_demo_20261025 import EventMeter
    from LightGenV2.tasks.t06_video_quality_assessment import lab_runtime as native
    fs=torch.cuda.Stream();bs=torch.cuda.Stream();ts=torch.cuda.Stream();origin=torch.cuda.Event(enable_timing=True)
    fm=EventMeter();bm=EventMeter();live.instrument_body(model,bm)
    tensor_queue=queue.Queue(maxsize=2);failure=[];cancel=threading.Event();events=[];predictions=[]
    origin.record();origin.synchronize();start_wall=time.perf_counter()
    def produce():
        try:
            with torch.inference_mode(),torch.cuda.stream(fs):
                for g in range(TOTAL//16):
                    guard.fast()
                    if cancel.is_set():return
                    fm.field=g;lo=len(fm.rows);a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
                    batch=native_lg_front(front,position,raw[g%(UNIQUE//16)],fm,ts);b.record()
                    item=(g,batch,a,b,lo,len(fm.rows))
                    while not cancel.is_set():
                        try:tensor_queue.put(item,timeout=.25);break
                        except queue.Full:guard.fast()
        except BaseException as e:failure.append(e);cancel.set()
    thread=None
    try:
        if mode=='pipeline':thread=threading.Thread(target=produce,daemon=True);thread.start()
        pending=deque()
        with torch.inference_mode():
            for g in range(TOTAL//16):
                guard.fast()
                if mode=='pipeline':
                    while True:
                        if failure:raise failure[0]
                        try:item=tensor_queue.get(timeout=.25);break
                        except queue.Empty:guard.fast()
                else:
                    with torch.cuda.stream(fs):
                        fm.field=g;lo=len(fm.rows);a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
                        batch=native_lg_front(front,position,raw[g%(UNIQUE//16)],fm,ts);b.record();item=(g,batch,a,b,lo,len(fm.rows))
                index,batch,a,b,lo,hi=item
                if index!=g:raise RuntimeError('Field order changed')
                with torch.cuda.stream(bs):
                    bs.wait_event(b);bm.field=g;body_lo=len(bm.rows)
                    c=torch.cuda.Event(enable_timing=True);d=torch.cuda.Event(enable_timing=True);c.record()
                    prediction=native.forward(model,batch)['prediction'];d.record()
                pending.append((d,batch));predictions.append(prediction)
                events.append(dict(field=g,front_start=a,front_end=b,body_start=c,body_end=d,front_rows=(lo,hi),body_rows=(body_lo,len(bm.rows))))
                if mode=='serial':d.synchronize()
                elif len(pending)>2:pending.popleft()[0].synchronize()
                if (g+1)%16==0:write(folder/'status.json',dict(status='GPU_timing',mode=mode,completed_video_calls=(g+1)*16,total=TOTAL))
            events[-1]['body_end'].synchronize();ended_wall=time.perf_counter()
        if thread:thread.join(timeout=10)
        if failure:raise failure[0]
        torch.cuda.synchronize()
        front_stages=fm.records(0);body_stages=bm.records(0);records=[]
        for ev in events:
            row={k:origin.elapsed_time(ev[k]) for k in ('front_start','front_end','body_start','body_end')};row['field']=ev['field']
            row['front_stages']=front_stages[slice(*ev['front_rows'])];row['body_stages']=body_stages[slice(*ev['body_rows'])]
            if sum(s['family']=='optical_simulation_FFT' for s in row['body_stages'])!=6:raise RuntimeError('Not six full-field optical passes')
            records.append(row)
        overlap_ms=sum(overlap([(r['front_start'],r['front_end']) for r in records if r['field']>i],r['body_start'],r['body_end']) for i,r in enumerate(records))
        end=records[-1]['body_end'];begin=records[0]['front_start'];elapsed=end-begin
        array=torch.cat(predictions,dim=0).float().cpu().numpy()
        if array.shape!=(TOTAL//16,16) or not np.isfinite(array).all():raise RuntimeError('Wrong full-field outputs')
        for c in range(1,CYCLES):
            if not np.array_equal(array[:32],array[c*32:(c+1)*32]):raise RuntimeError('Repeated input output changed')
        np.savez_compressed(folder/(mode+'_outputs.npz'),prediction=array)
        first=records[0]['body_end']-begin
        info=dict(status='complete',mode=mode,video_calls=TOTAL,unique_videos=UNIQUE,fields=TOTAL//16,frames_per_field=64,
            first_GPU_front_start_ms=begin,last_GPU_score_ms=end,continuous_GPU_timeline_ms=elapsed,
            dispatch_to_last_score_host_ms=(ended_wall-start_wall)*1000,time16_ms=elapsed*16/TOTAL,equivalent_frame_ms=elapsed/(TOTAL*4),
            actual_later_front_overlapping_body_ms=overlap_ms,
            prefixes=[dict(video_calls=n,time16_ms=(records[n//16-1]['body_end']-begin)*16/n) for n in prefixes()],
            first_group_ms=first,steady=steady_report([r['body_end']-begin for r in records]),records=records,
            actual_complete_software_includes_FFT=True,physical_optical_E2E_measured=False,
            fixed_optical_ms_per_field_reference=6*(1000/1400+.3+.0307),
            no_dummy_warmup=True,no_fast_repeat_selection=True,model_math_unchanged=True)
        write(folder/(mode+'_timing.json'),info);return array
    finally:
        cancel.set()
        if thread:thread.join(timeout=10)
        bm.restore()


class QwenTimer:
    def __init__(self,model):
        self.records=[];self.state=None;self.handles=[]
        def before(owner,inputs):
            import torch
            event=torch.cuda.Event(enable_timing=True);event.record();self.state['first_TF']=event
        self.handles.append(model.model.visual.blocks[0].register_forward_pre_hook(before))
    def reset(self,expected_calls=1):
        import torch
        a=torch.cuda.Event(enable_timing=True);a.record();self.state=dict(start=a,host_start=time.perf_counter())
    def finish(self):
        import torch
        b=torch.cuda.Event(enable_timing=True);b.record();b.synchronize();self.state.update(end=b,host_end=time.perf_counter())
        if 'first_TF' not in self.state:raise RuntimeError('Original first visual TF absent')
        self.records.append(self.state)
        return dict(vision_first_block_to_score_cuda_ms=self.state['first_TF'].elapsed_time(b),vision_first_block_to_score_host_ms=1000*(self.state['host_end']-self.state['host_start']))
    def close(self):
        for h in self.handles:h.remove()


def qwen_worker(args,folder,guard):
    import torch,numpy as np
    import lgvq_live_frontend as live
    from LightGenV2.tasks.t06_video_quality_assessment import quality_token_batch_benchmark_200 as native
    reference=live.TASK/'runs/simulation/lgvq_mps_e2e_a100_20261024/lab_remaining01/qwen50/qwen_cap50_repeat01/identity.json'
    identity=json.loads(reference.read_text());cmd=identity['command'];value=lambda flag:cmd[cmd.index(flag)+1]
    checkpoint=Path(value('--checkpoint'))
    if sha(checkpoint)!=identity['expected_head_sha256']:raise RuntimeError('Original Qwen head changed')
    opt=SimpleNamespace(model=Path(value('--model-path')),checkpoint=checkpoint,manifest=live.MANIFEST,image_size=448)
    for name,digest in identity['assets'].items():
        if sha(opt.model/name)!=digest:raise RuntimeError('Original Qwen asset differs: '+name)
    processor,model,head,scores,rows,prompt=native.load_runtime(opt);guard.bind()
    if torch.cuda.get_device_properties(0).multi_processor_count!=54:raise RuntimeError('Qwen50 must expose54SM')
    times=[];tokens=[];processor.tokenizer=TokenizerAudit(processor.tokenizer,tokens);inputs=[]
    for i,row in enumerate(rows[:UNIQUE]):
        guard.fast();a=time.perf_counter();cpu,pos=native.prepare_batch([row],processor,prompt,torch.device('cpu'),448);b=time.perf_counter()
        device={k:v.to('cuda') for k,v in cpu.items()};torch.cuda.synchronize();c=time.perf_counter()
        inputs.append(device);times.append(dict(index=i,sample_id=row['sample_id'],CPU_start=a,CPU_end=b,H2D_end=c,positions=pos))
        if (i+1)%16==0:write(folder/'status.json',dict(status='preparing_original_Qwen_inputs',completed_unique=i+1,total_unique=UNIQUE))
    write(folder/'external_preparation.json',dict(rows=times,tokenizer_calls=tokens,raw_inputs_GPU_resident=True,neural_features_cached=False,excluded_from_GPU_window=True))
    timer=QwenTimer(model);values=[]
    try:
        with torch.inference_mode():
            for i,index,cycle in plan():
                guard.fast();_,pred,_=native.forward_batch(inputs[index],model,head,scores,timer);values.append(pred[0])
                if (i+1)%128==0:write(folder/'status.json',dict(status='Qwen50_serial_GPU_timing',completed_video_calls=i+1,total=TOTAL))
        a=timer.records[0]['start'];rows_out=[]
        for i,rec in enumerate(timer.records):rows_out.append(dict(index=i,source_index=i%UNIQUE,cycle=i//UNIQUE,start_ms=a.elapsed_time(rec['start']),first_TF_ms=a.elapsed_time(rec['first_TF']),end_ms=a.elapsed_time(rec['end']),host_start=rec['host_start'],host_end=rec['host_end']))
        elapsed=rows_out[-1]['end_ms'];host=(timer.records[-1]['host_end']-timer.records[0]['host_start'])*1000
        array=np.asarray(values,dtype=np.float32).reshape(CYCLES,UNIQUE)
        if not all(np.array_equal(array[0],v) for v in array[1:]):raise RuntimeError('Qwen repeated outputs differ')
        np.savez_compressed(folder/'qwen_outputs.npz',prediction=array)
        write(folder/'qwen_timing.json',dict(status='complete',video_calls=TOTAL,unique_videos=UNIQUE,cap=50,batch_videos=1,
            continuous_GPU_timeline_ms=elapsed,continuous_host_ms=host,time16_ms=elapsed*16/TOTAL,
            all_inter_video_waits_included=True,terminal_CPU_copy_excluded=True,no_dummy_warmup=True,records=rows_out,
            prefixes=[dict(video_calls=n,time16_ms=rows_out[n-1]['end_ms']*16/n) for n in prefixes()],
            steady=steady_report([rows_out[i]['end_ms'] for i in range(15,TOTAL,16)]),original_forward_function_reused=True))
        for name,digest in identity['assets'].items():
            if sha(opt.model/name)!=digest:raise RuntimeError('Qwen asset changed during timing: '+name)
        return [dict(sample_id=r['sample_id'],video_path=str(r['video_path'])) for r in rows[:UNIQUE]]
    finally:timer.close()


def worker(args):
    import torch,numpy as np
    import lgvq_live_frontend as live
    from lgvq_pipeline_demo_20261025 import PreparedFront,compare_inputs
    from LightGenV2.tasks.t06_video_quality_assessment import lab_runtime as native
    from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.data import read_manifest
    root=args.output;cap=100 if args.system=='lg' else 50
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU or os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE')!=str(cap):raise RuntimeError('Physical GPU/MPS binding required')
    if root.exists():raise FileExistsError(root)
    root.mkdir();guard=model_guard(args.deadline,cap);torch.set_num_threads(8);before=source_map()
    identity=dict(profile='GPU_model_input_resident_512x4_pipeline_20261026_v1',system=args.system,repeat=args.repeat,
        gpu_uuid=GPU,cap=cap,pid=os.getpid(),sources=before,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        published_measurement_commit=os.environ.get('MEASUREMENT_PUBLISHED_COMMIT'),
        source_installation='scoped published Git paths; runtime HEAD separately recorded; peer training source not changed',
        unique_videos=UNIQUE,video_calls=TOTAL,sequence='first512TEST in original order, repeated4 times',
        primary_start='first GPU model frontend',primary_end='last score ready on GPU',all_internal_gaps_retained=True,
        external_preparation_recorded_not_charged=True,no_input_embedding_or_prediction_cache=True,
        tokenizer_is_CPU_separate_not_added_to_GPU_timeline=True,no_training=True,no_full_TEST_accuracy_evaluation=True)
    write(root/'identity.json',identity);write(root/'status.json',dict(status='loading',pid=os.getpid()))
    try:
        if sha(live.MANIFEST)!=live.MANIFEST_SHA:raise RuntimeError('Original manifest changed')
        if args.system=='lg':
            tests=[r for r in read_manifest(live.MANIFEST) if r.split=='test'];front=live.LiveFront('cuda:0')
            model,_=native.load_model('temporal',live.TASK/'runs/simulation/multivideo16x4_rank_s163/best_checkpoint.pt','cuda:0');model.requires_grad_(False)
            guard.bind()
            if torch.cuda.get_device_properties(0).multi_processor_count!=108:raise RuntimeError('LG100 requires108SM')
            fixed=PreparedFront(front);position=fixed.position4.repeat(16,1)
            inputs,files=prep_lg(front,tests,root,guard)
            order=('serial','pipeline') if args.repeat%2 else ('pipeline','serial');arrays={}
            for mode in order:arrays[mode]=lg_timed(model,front,position,inputs,guard,mode,root)
            if not np.array_equal(arrays['serial'],arrays['pipeline']):raise RuntimeError('LG serial/pipeline full outputs differ')
            # Reference verification AFTER all timed forwards; never a warmup.
            with torch.inference_mode():
                original,meta=front.build(tests[:16],'temporal');expected=native.forward(model,original)['prediction'].float().cpu().numpy()
            if not np.array_equal(expected,arrays['pipeline'][:1]):raise RuntimeError('Original native full field differs')
            write(root/'consistency.json',dict(status='passed',validation_after_timing=True,serial_pipeline_all_outputs_equal=True,
                original_first_full_field_output_equal=True,reference_sample_ids=meta['source_ids'],geometry_position_setup_s=fixed.setup_position_s))
        else:files=qwen_worker(args,root,guard)
        for row in files:row['bytes']=Path(row['video_path']).stat().st_size;row['sha256']=sha(row['video_path'])
        write(root/'input_files.json',files)
        if source_map()!=before:raise RuntimeError('Frozen runtime source changed')
        guard.check();write(root/'status.json',dict(status='complete',pid=os.getpid(),video_calls=TOTAL,finished=time.time()))
    except BaseException as e:write(root/'status.json',dict(status='failed',error=repr(e),pid=os.getpid()));raise
    finally:guard.close(root/'process_audit.json')


def suite(args):
    import fcntl
    from first_tf_supplement_20261025 import serialize_mps_commands
    from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
    from LightGenV2.tasks.t06_video_quality_assessment.measure_mps_dataset_once import validate_idle_memory
    import run_temporal_remainder as loading
    serialize_mps_commands();engine.GPU=GPU;engine.PIPE=PIPE;engine.ROOT=args.output
    if args.output.exists() or not args.output.is_relative_to(ROOT) or not 0<args.deadline-time.time()<=10800:raise ValueError('New bounded local-lab root required')
    args.output.mkdir();state=dict(status='admission',pid=os.getpid(),deadline=args.deadline,completed=[]);child=None;active=False
    def save():write(args.output/'suite_status.json',state)
    def bounded():
        if time.time()>args.deadline-60:raise TimeoutError('Finite authorized run')
    save()
    try:
        with (ROOT/'.temporal_telemetry_a100.lock').open('a') as lock:
            while True:
                bounded()
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:time.sleep(5)
            for system in ('lg','qwen'):
                for repeat in (1,2,3):
                    for idle in range(12):
                        bounded()
                        if engine.base.guard_records(GPU):raise RuntimeError('Foreign A100 before admission; no load')
                        validate_idle_memory(GPU);state.update(status='idle_guard',system=system,repeat=repeat,idle_checks=idle+1);save();time.sleep(10)
                    old=subprocess.run(['nvidia-cuda-mps-control'],input='get_server_list\n',text=True,capture_output=True,env=dict(os.environ,CUDA_MPS_PIPE_DIRECTORY=PIPE),timeout=5)
                    if old.returncode==0:raise RuntimeError('Private MPS not released')
                    engine.daemon(True);active=True;cap=100 if system=='lg' else 50
                    leaf=args.output/f'{system}_repeat{repeat:02d}'
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_MPS_PIPE_DIRECTORY=PIPE,CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=str(cap),
                        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
                    cmd=['/home/guest3/miniconda3/envs/xml/bin/python','-u',str(Path(__file__)),'--phase','worker','--system',system,'--repeat',str(repeat),'--output',str(leaf),'--deadline',str(args.deadline)]
                    observations=[]
                    with leaf.with_suffix('.log').open('x') as log:
                        child=subprocess.Popen(cmd,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        state.update(status='running',child=child.pid,system=system,repeat=repeat);save()
                        try:
                            while child.poll() is None:
                                bounded()
                                snap=loading.process_snapshot(GPU,child.pid,PIPE)
                                observations.append(snap)
                                if snap['foreign_gpu_processes'] or snap['foreign_mps_clients']:raise RuntimeError('Foreign A100 during authorized run')
                                time.sleep(2)
                        except BaseException:
                            if child.poll() is None:os.killpg(child.pid,signal.SIGINT);child.wait(timeout=45)
                            raise
                    write(args.output/f'{system}_repeat{repeat:02d}_loading_and_process_guard.json',observations)
                    engine.daemon(False);active=False
                    if child.returncode:raise RuntimeError('Worker failed; preserve raw and diagnose, no blind rerun')
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
    p.add_argument('--system',choices=['lg','qwen']);p.add_argument('--repeat',type=int);p.add_argument('--output',type=Path,required=True);p.add_argument('--deadline',type=float,required=True)
    a=p.parse_args();a.output=a.output.resolve();(suite if a.phase=='suite' else worker)(a)
