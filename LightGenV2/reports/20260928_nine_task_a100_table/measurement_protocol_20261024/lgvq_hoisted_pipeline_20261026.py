"""LG-only paired experiment: hoist ORIGINAL input adapters, tokenize online.

No Qwen execution, no new model math, no feature/prediction cache. The original
forward AST is partitioned at the routing declaration: input adapters run on the
next-field frontend stream, and the EXACT remaining original AST runs on the
body stream. Preserve full-field six-pass propagation and every measured field.
"""
import argparse
import ast
from collections import deque
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import textwrap
import threading
import time

import lgvq_minimal_pipeline_20261026 as minimal
import qwen_shared_serial_20261026 as shared

base=shared.base
PREVIOUS_SOURCES=base.source_map
PREVIOUS_ENTRY=Path(base.__file__)
base.__file__=__file__
base.PIPE='/DATA/DATA1/guest3/.cache/lgvq_hoisted_pipeline_20261026'
base.model_guard=shared.SharedGuard
PREFIX_NAMES=('batch','videos','vision','prompt_tokens','prompt_mask')


def sources():
    result=PREVIOUS_SOURCES()
    for p in (PREVIOUS_ENTRY,Path(minimal.__file__),Path(__file__)):
        result[p.relative_to(base.REPO).as_posix()]=base.sha(p)
    return result


base.source_map=sources


def partition_forward(source):
    """Pure AST partition; do not rewrite/simplify a single original tail op."""
    node=ast.parse(textwrap.dedent(source)).body[0]
    assert isinstance(node,ast.FunctionDef) and node.name=='forward'
    boundaries=[i for i,n in enumerate(node.body) if isinstance(n,ast.AnnAssign)
                and isinstance(n.target,ast.Name) and n.target.id=='routing']
    assert len(boundaries)==1,'Pinned routing boundary required'
    cut=boundaries[0]
    prefix=copy.deepcopy(node);prefix.name='prepare_native_prefix';prefix.decorator_list=[]
    prefix.body=copy.deepcopy(node.body[:cut])+[
        ast.Return(value=ast.Dict(keys=[ast.Constant(x) for x in PREFIX_NAMES],
                                 values=[ast.Name(id=x,ctx=ast.Load()) for x in PREFIX_NAMES]))]
    tail=ast.parse('def forward_from_prefix(self,prefix,*,optical_enabled=True):\n    pass').body[0]
    bindings=ast.parse('\n'.join(f'{x}=prefix[{x!r}]' for x in PREFIX_NAMES)).body
    tail.body=bindings+copy.deepcopy(node.body[cut:])
    assert ast.dump(ast.Module(body=tail.body[len(bindings):],type_ignores=[]))==ast.dump(ast.Module(body=node.body[cut:],type_ignores=[]))
    module=ast.fix_missing_locations(ast.Module(body=[prefix,tail],type_ignores=[]))
    return module,dict(original_forward_sha256=hashlib.sha256(source.encode()).hexdigest(),
        original_tail_AST_sha256=hashlib.sha256(ast.dump(ast.Module(body=node.body[cut:],type_ignores=[])).encode()).hexdigest(),
        prefix_statement_count=cut,original_tail_statement_count=len(node.body)-cut,
        original_tail_AST_unchanged=True,prefix_values=list(PREFIX_NAMES))


def original_functions(model):
    original=type(model).forward
    module,receipt=partition_forward(inspect.getsource(original))
    namespace=dict(original.__globals__)
    exec(compile(module,inspect.getsourcefile(original)+'::partitioned', 'exec'),namespace)
    return namespace['prepare_native_prefix'],namespace['forward_from_prefix'],receipt


def optimized_timed(model,front,position,raw,guard,folder):
    import numpy as np
    import torch
    from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54 import cache_qwen_front as native_front
    prefix_fn,tail_fn,partition=original_functions(model)
    # Geometry/dtype-only constant; no sample-dependent neural result cached.
    position=position.to(dtype=torch.bfloat16)
    reference=json.loads((folder.parent/'external_preparation.json').read_text())['rows'][0]
    fs,bs,ts=[torch.cuda.Stream() for _ in range(3)]
    origin=torch.cuda.Event(enable_timing=True);origin.record();origin.synchronize()
    tensor_queue=queue.Queue(maxsize=2);cancel=threading.Event();errors=[]
    records=[];outputs=[];token_records=[];context={};handles=[]
    for name in ('parallel_router','parallel_optics','serial_router','serial_optics'):
        def count(*args):context['passes']=context.get('passes',0)+1
        handles.append(getattr(model,name).propagation.register_forward_pre_hook(count))
    def first_optics(*args):
        if 'first' not in context:
            event=torch.cuda.Event(enable_timing=True);event.record();context['first']=event
    handles.append(model.parallel_optics.width_to_field.register_forward_pre_hook(first_optics))
    host_start=time.perf_counter()

    def prepare(g):
        original=raw[g%32]
        with torch.cuda.stream(fs):
            a=torch.cuda.Event(enable_timing=True);a.record()
        # Actual tokenizer on EVERY field, inside the continuous online window.
        started=time.perf_counter()
        rendered=native_front.render_prompt(front.tokenizer,target_name='temporal',prompt=native_front.TARGET_PROMPTS['temporal'])
        tok=front.tokenizer([rendered],padding=True,return_tensors='pt',add_special_tokens=False)
        ended=time.perf_counter()
        token_records.append(dict(field=g,start=started,end=ended,input_ids=tok['input_ids'].tolist(),mask=tok['attention_mask'].tolist()))
        ids_cpu=tok['input_ids'].pin_memory();mask_cpu=tok['attention_mask'].bool().pin_memory()
        with torch.cuda.stream(ts):
            ts.wait_event(a)
            ids=ids_cpu.to('cuda',non_blocking=True);mask=mask_cpu.to('cuda',non_blocking=True)
        current=dict(original,ids=ids,mask=mask)
        with torch.cuda.stream(fs):
            batch=base.native_lg_front(front,position,current,minimal.NoInnerTimer(),ts)
            prefix=prefix_fn(model,**batch,optical_enabled=True)
            b=torch.cuda.Event(enable_timing=True);b.record()
        return g,prefix,a,b,(batch,ids_cpu,mask_cpu,ids,mask)

    def producer():
        try:
            with torch.inference_mode():
                for g in range(base.TOTAL//16):
                    if cancel.is_set():return
                    guard.fast();item=prepare(g)
                    while not cancel.is_set():
                        try:tensor_queue.put(item,timeout=.25);break
                        except queue.Full:guard.fast()
        except BaseException as error:errors.append(error);cancel.set()

    thread=threading.Thread(target=producer,daemon=True);pending=deque()
    try:
        thread.start()
        with torch.inference_mode():
            for g in range(base.TOTAL//16):
                guard.fast()
                while True:
                    if errors:raise errors[0]
                    try:item=tensor_queue.get(timeout=.25);break
                    except queue.Empty:guard.fast()
                index,prefix,a,b,keepalive=item;assert index==g
                with torch.cuda.stream(bs):
                    bs.wait_event(b);context.clear()
                    c=torch.cuda.Event(enable_timing=True);d=torch.cuda.Event(enable_timing=True);c.record()
                    prediction=tail_fn(model,prefix,optical_enabled=True)['prediction'];d.record()
                    assert context.get('passes')==6 and 'first' in context
                    records.append(dict(field=g,front_start=a,front_end=b,body_start=c,first_optic=context['first'],body_end=d,optical_passes=6))
                outputs.append(prediction);pending.append((d,prefix,keepalive))
                if len(pending)>2:pending.popleft()[0].synchronize()
                if (g+1)%16==0:base.write(folder/'status.json',dict(status='online_tokenizer_and_hoisted_adapters',completed_video_calls=(g+1)*16,total=base.TOTAL))
            records[-1]['body_end'].synchronize();host_end=time.perf_counter()
        thread.join(10)
        if errors:raise errors[0]
        assert not thread.is_alive() and len(token_records)==128
        assert all(r['input_ids']==reference['input_ids'] and r['mask']==reference['mask'] for r in token_records)
        torch.cuda.synchronize()
        rows=[dict(field=r['field'],optical_passes=6,**{k:origin.elapsed_time(r[k]) for k in ('front_start','front_end','body_start','first_optic','body_end')}) for r in records]
        begin=rows[0]['front_start'];elapsed=rows[-1]['body_end']-begin
        array=torch.cat(outputs).float().cpu().numpy()
        assert array.shape==(128,16) and np.isfinite(array).all()
        assert all(np.array_equal(array[:32],array[32*c:32*(c+1)]) for c in (1,2,3))
        np.savez_compressed(folder/'pipeline_outputs.npz',prediction=array)
        ov=sum(base.overlap([(r['front_start'],r['front_end']) for r in rows[i+1:]],a['body_start'],a['body_end']) for i,a in enumerate(rows))
        base.write(folder/'pipeline_timing.json',dict(status='complete',mode='online_tokenizer_hoisted_adapters',
            video_calls=2048,unique_videos=512,fields=128,frames_per_field=64,
            first_GPU_front_start_ms=begin,last_GPU_score_ms=rows[-1]['body_end'],continuous_GPU_timeline_ms=elapsed,
            dispatch_to_last_score_host_ms=(host_end-host_start)*1000,time16_ms=elapsed/128,equivalent_frame_ms=elapsed/8192,
            CPU_inclusive_host_primary_ms=(host_end-host_start)*1000,
            host_primary_start='producer dispatch before first online tokenizer',
            GPU_timeline_start='first frontend CUDA event; host-leading work separately retained',
            records=rows,first_group_ms=rows[0]['body_end']-begin,
            prefixes=[dict(video_calls=n,time16_ms=(rows[n//16-1]['body_end']-begin)*16/n) for n in base.prefixes()],
            steady=base.steady_report([r['body_end']-begin for r in rows]),
            actual_later_front_overlapping_body_ms=ov,
            actual_complete_software_includes_FFT=True,physical_optical_E2E_measured=False,
            online_CPU_tokenizer_in_continuous_window=True,online_tokenizer_field_calls=128,
            input_adapters_moved_from_body_to_future_field_front=True,original_tail_AST_unchanged=True,
            no_embedding_or_prediction_cache=True,no_dummy_warmup=True,all_fields_and_internal_gaps_retained=True))
        base.write(folder/'online_tokenizer.json',dict(records=token_records,CPU_work_sum_ms=sum((r['end']-r['start'])*1000 for r in token_records),
            all_actual_IDs_and_masks_equal_original=True,not_to_be_added_twice_to_continuous_window=True))
        base.write(folder/'AST_partition.json',partition)
        # Full original output tree, after timing, no validation warmup.
        with torch.inference_mode():
            batch=base.native_lg_front(front,position,raw[0],minimal.NoInnerTimer(),torch.cuda.current_stream())
            original=model(**batch,optical_enabled=True)
            altered=tail_fn(model,prefix_fn(model,**batch,optical_enabled=True),optical_enabled=True)
            from lgvq_live_frontend import equal_tree
            equal_tree(original,altered)
        base.write(folder/'full_output_equivalence.json',dict(status='passed',original_vs_partitioned_full_tree_bitwise_equal=True,validation_after_timing=True))
        return array
    finally:
        cancel.set();thread.join(10)
        for h in handles:h.remove()


def paired_timed(model,front,position,raw,guard,mode,folder):
    name='control' if mode=='serial' else 'hoisted'
    child=folder/name;child.mkdir()
    base.write(folder/'status.json',dict(status='paired_LG_GPU_measurement',strategy=name))
    if mode=='serial':return minimal.timed(model,front,position,raw,guard,'pipeline',child)
    return optimized_timed(model,front,position,raw,guard,child)


base.lg_timed=paired_timed


def worker(args):
    assert args.system=='lg'
    base.worker(args)
    p=args.output/'identity.json';identity=json.loads(p.read_text())
    identity.update(profile='LG_paired_original_pipeline_vs_online_tokenizer_hoisted_adapters_20261026_v1',
        nonexclusive_authorized=True,foreign_GPU_processes_observed_not_rejected=True,
        not_uncontended_baseline_performance=True,private_mps_pipe=base.PIPE,
        paired_order=['control','hoisted'] if args.repeat%2 else ['hoisted','control'],
        both_strategies_recompute_all_neural_features=True,CPU_tokenizer_online_only_in_hoisted=True,
        CPU_tokenizer_scope=dict(control='before primary',hoisted='online producer; leading work included in host primary'),
        Qwen_not_executed=True)
    identity.pop('tokenizer_is_CPU_separate_not_added_to_GPU_timeline',None)
    base.write(p,identity)


def suite(args):
    import fcntl
    from first_tf_supplement_20261025 import serialize_mps_commands
    from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
    from run_temporal_remainder import process_snapshot
    serialize_mps_commands();engine.GPU=base.GPU;engine.PIPE=base.PIPE;engine.ROOT=args.output
    assert not args.output.exists() and args.output.is_relative_to(base.ROOT) and 0<args.deadline-time.time()<=14400
    args.output.mkdir();state=dict(status='admission',pid=os.getpid(),deadline=args.deadline,completed=[],Qwen_not_executed=True)
    active=False;child=None
    def save():base.write(args.output/'suite_status.json',state)
    def bounded():
        if time.time()>args.deadline-60:raise TimeoutError('Finite LG optimization experiment')
    save()
    try:
        with (base.ROOT/'.temporal_telemetry_a100.lock').open('a') as lock:
            while True:
                bounded()
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:time.sleep(2)
            previous=json.loads((args.previous/'suite_status.json').read_text())
            assert previous['status']=='complete' and previous['mps_released'] and not Path('/proc',str(previous['pid'])).exists()
            state['retained_Qwen_root']=str(args.previous);save()
            for repeat in (1,2,3):
                bounded()
                free=float(subprocess.check_output(['nvidia-smi','-i',base.GPU,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
                assert free>=8192,'Insufficient free memory'
                assert shared.control('get_server_list').returncode!=0,'Own private MPS already active'
                engine.daemon(True);active=True;leaf=args.output/f'lg_repeat{repeat:02d}'
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=base.GPU,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_MPS_PIPE_DIRECTORY=base.PIPE,
                    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE='100',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',
                    MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
                command=['/home/guest3/miniconda3/envs/xml/bin/python','-u',str(Path(__file__)),'--phase','worker','--system','lg',
                    '--repeat',str(repeat),'--output',str(leaf),'--deadline',str(args.deadline)]
                observations=[]
                try:
                    with leaf.with_suffix('.log').open('x') as log:
                        child=subprocess.Popen(command,cwd=base.REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        state.update(status='running',repeat=repeat,child=child.pid,free_memory_mib_before_loading=free);save()
                        while child.poll() is None:
                            bounded();snapshot=process_snapshot(base.GPU,child.pid,base.PIPE)
                            shared.validate_shared_snapshot(snapshot,child.pid);observations.append(snapshot);time.sleep(2)
                except BaseException:
                    if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGINT);child.wait(timeout=45)
                    raise
                finally:base.write(args.output/f'lg_repeat{repeat:02d}_loading_and_process_guard.json',observations)
                shared.stop_private_daemon();active=False
                if child.returncode:raise RuntimeError('LG experiment failed; preserve raw before missing-only continuation')
                state['completed'].append(leaf.name);save()
            state.update(status='complete',finished=time.time())
    except BaseException as error:state.update(status='stopped',error=repr(error));raise
    finally:
        if active:
            try:shared.stop_private_daemon();state['mps_released']=True
            except BaseException as error:state['mps_release_error']=repr(error)
        else:state['mps_released']=True
        save()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['suite','worker'],required=True)
    parser.add_argument('--system',choices=['lg'],default='lg');parser.add_argument('--repeat',type=int)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--previous',type=Path)
    parser.add_argument('--deadline',type=float,required=True)
    options=parser.parse_args();options.output=options.output.resolve()
    (suite if options.phase=='suite' else worker)(options)
