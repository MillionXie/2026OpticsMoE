"""One host launch for the unchanged ordered body graphs, all timestamps kept.

Composition/initialization occurs inside the continuous measured window. The
two original real captures and all first three eager fields remain unchanged.
CUDA external event-record nodes retain per-segment original device clocks;
no phase duration, queue gap, failed output or cold-start cost is subtracted.
"""
import ast
import ctypes
import hashlib
import inspect
from pathlib import Path
import time
import lgvq_async_replay_20261026 as original


def worker_AST(source):
    tree=ast.parse(source);changes=[]
    class Imports(ast.NodeTransformer):
        def visit_Import(self,node):
            if len(node.names)==1 and node.names[0].name=='lgvq_async_replay_20261026' and node.names[0].asname=='replay':
                changes.append('replay');node.names[0].name='lgvq_a100_batch_replay_20261027'
            return node
        def visit_ImportFrom(self,node):
            if node.module=='lgvq_async_raw_adapter_20261026' and [(a.name,a.asname) for a in node.names]==[('adapt',None)]:
                changes.append('adapter');node.module='lgvq_a100_batch_raw_adapter_20261027'
            return node
    tree=Imports().visit(tree)
    if sorted(changes)!=['adapter','replay']:raise ValueError('Exact original worker import boundaries required')
    return ast.fix_missing_locations(tree),dict(original_worker_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        only_two_imports_redirected=True,first_three_eager_and_two_real_captures_unchanged=True)


class ExternalRecorder:
    """Bind only the already-loaded exact CUDA runtime; no new CUDA context."""
    def __init__(self):
        paths={line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
               if 'libcudart.so.' in line and line.split()[-1].startswith('/')}
        if len(paths)!=1:raise RuntimeError('One already loaded CUDA runtime required')
        self.path=next(iter(paths));self.lib=ctypes.CDLL(self.path)
        self.fn=self.lib.cudaEventRecordWithFlags
        self.fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint];self.fn.restype=ctypes.c_int

    def record(self,event,stream):
        if not event.cuda_event:raise ValueError('Timing event must be initialized inside measured build')
        code=self.fn(event.cuda_event,stream.cuda_stream,1)  # cudaEventRecordExternal
        if code:raise RuntimeError('cudaEventRecordWithFlags failed: '+str(code))


def compose(torch,slot,stream,pool,recorder):
    if pool or len(slot['segments'])!=38 or not slot.get('phase_detail'):
        raise ValueError('Fresh original detailed slot required')
    for _ in range(80):pool.append(torch.cuda.Event(enable_timing=True))
    with torch.cuda.stream(stream):
        # Materialize actual event handles before external records, still timed.
        for e in pool:e.record()
        pool[-1].synchronize()
        graph=torch.cuda.CUDAGraph();graph.capture_begin(capture_error_mode='global')
        try:
            for j,segment in enumerate(slot['segments'][1:]):
                recorder.record(pool[4+2*j],stream)
                segment['graph'].replay()
                recorder.record(pool[5+2*j],stream)
        finally:graph.capture_end()
    return graph


def replay_fields(torch,slots,raw,tokenize,guard,front_stream,body_stream,*,first_field,total_fields,
                  reuse_events=False,snapshot_completed=None):
    original.validate_slots(slots)
    if (first_field!=5 or total_fields!=128 or len(raw)!=32 or not reuse_events or snapshot_completed is None):
        raise ValueError('Exact complete two-slot timed event-pool workload required')
    outputs=[];records=[];pools=[[],[]];graphs=[None,None];pending=[None,None];snapshots=0
    recorder=None;builds=[]
    with torch.inference_mode():
        # Both original real captures have finished. Compose BOTH independent
        # launch plans before starting the steady pipeline, never drain a new
        # field just to compose the second slot. This stays inside host_start/end.
        for index,slot in enumerate(slots):
            sync_start=time.perf_counter();torch.cuda.synchronize();sync_end=time.perf_counter()
            build_start=time.perf_counter()
            if recorder is None:recorder=ExternalRecorder()
            graphs[index]=compose(torch,slot,body_stream,pools[index],recorder)
            builds.append(dict(sync_start=sync_start,sync_end=sync_end,start=build_start,end=time.perf_counter(),
                slot=index,child_graphs=37,timing_event_pairs=37,event_record_flag='cudaEventRecordExternal',
                timed_initial_event_materializations=80,extra_model_forwards=0,
                all_composition_cost_charged=True,runtime_library=recorder.path))
        for field in range(first_field,total_fields):
            guard.fast();index=(field-first_field)%2;slot=slots[index]
            cpu_ids,cpu_mask,token_record=tokenize(field)
            if token_record['field']!=field:raise ValueError('Original tokenizer order changed')
            pin_start=time.perf_counter();ids=cpu_ids.pin_memory();mask=cpu_mask.bool().pin_memory();pin_end=time.perf_counter()
            wait_start=time.perf_counter();slot['score_ready'].synchronize();wait_end=time.perf_counter()
            if pending[index] is not None:
                prior=pending[index];snapshot_start=time.perf_counter()
                records[prior]=snapshot_completed(records[prior])
                records[prior]['event_snapshot_before_reuse']=dict(reused_at_field=field,
                    host_start=snapshot_start,host_end=time.perf_counter(),prior_score_synchronized=True)
                snapshots+=1
            pool=pools[index];observed=[];current=raw[field%32]
            with torch.cuda.stream(front_stream):
                h0,h1,x,y=pool[:4]
                h2d_start=time.perf_counter();h0.record()
                slot['static']['ids'].copy_(ids,non_blocking=True);slot['static']['mask'].copy_(mask,non_blocking=True)
                h1.record();h2d_end=time.perf_counter()
                submit_start=time.perf_counter();x.record()
                for name in ('pixels','quality'):slot['static'][name].copy_(current[name],non_blocking=True)
                slot['segments'][0]['graph'].replay();y.record()
                observed.append(dict(kind='frontend',label='input_refresh_and_original_prefix',start=x,end=y,
                    stream='frontend',host_submit_start=submit_start,host_submit_end=time.perf_counter()))
            with torch.cuda.stream(body_stream):
                body_stream.wait_event(y)
                batch_start=time.perf_counter();graphs[index].replay();batch_end=time.perf_counter()
                for j,segment in enumerate(slot['segments'][1:]):
                    observed.append(dict(kind=segment['kind'],label=segment['label'],start=pool[4+2*j],end=pool[5+2*j],
                        stream='body',host_submit_start=batch_start,host_submit_end=batch_end,
                        host_submission_owner='one_complete_body_graph_launch',external_timing_record=True))
                a,b=pool[78:80];submit_start=time.perf_counter();a.record()
                prediction=slot['prediction'].clone();b.record()
                observed.append(dict(kind='electronics',label='retain_current_score',start=a,end=b,stream='body',
                    host_submit_start=submit_start,host_submit_end=time.perf_counter()))
            slot['score_ready']=b;slot['keepalive']=(ids,mask);outputs.append(prediction)
            row=dict(field=field,slot=index,tokenizer=token_record,pin_host_start=pin_start,pin_host_end=pin_end,
                H2D_host_submit_start=h2d_start,H2D_host_submit_end=h2d_end,
                slot_reuse_wait_host_start=wait_start,slot_reuse_wait_host_end=wait_end,
                original_H2D_events=(h0,h1),events=observed,score_ready=b,host_submission_end=time.perf_counter(),
                body_batch_submission=dict(start=batch_start,end=batch_end,ordered_child_graphs=37,
                    external_device_event_pairs=37,one_host_call=True,all_shared_launch_cost_charged=True))
            if field==5:row['body_batch_builds']=builds
            records.append(row);pending[index]=len(records)-1
        final_start=time.perf_counter();records[-1]['score_ready'].synchronize();end_host=time.perf_counter()
        records[-1]['final_synchronization_observation']=dict(start=final_start,end=end_host)
    return dict(outputs=outputs,records=records,last_score_ready_host=end_host,
        event_reuse=dict(enabled=True,allocated_events=sum(map(len,pools)),completed_fields_snapshotted_before_reuse=snapshots,
            initialization_and_snapshot_overhead_inside_measured_window=True),
        actual_online_tokenizer_calls=123,independent_input_buffers=2,body_stream_serial=True,
        next_field_frontend_stream=True,all_startup_costs_owned_by_caller=True)
