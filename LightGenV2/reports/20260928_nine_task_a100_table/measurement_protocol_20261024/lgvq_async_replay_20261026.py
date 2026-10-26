"""Actual two-stream replay core; not yet a standalone qualified GPU worker.

The caller must capture two INDEPENDENT input/prefix/output allocations using
real retained fields, retain every eager/capture cost, and validate all scores.
This module only supplies the bounded online steady execution, with real
tokenization on every field and no feature/prediction/CCD cache.
"""
import time


def allocation_pointer(tensor):
    # Views may have different data_ptr() while still sharing one allocation.
    # Treat the entire storage as owned so an offset view cannot bypass checks.
    storage = getattr(tensor, 'untyped_storage', None)
    return storage().data_ptr() if storage is not None else tensor.data_ptr()


def validate_slots(slots):
    if len(slots)!=2:raise ValueError('Exactly two independent slots required')
    seen=set()
    for slot in slots:
        if not slot['segments'] or slot['segments'][0]['kind']!='frontend':
            raise ValueError('Separate frontend graph required')
        kinds=[x['kind'] for x in slot['segments']]
        if kinds.count('frontend')!=1 or kinds.count('optical_simulation')!=6 or kinds.count('electronics')!=(21 if slot.get('phase_detail') else 17):
            raise ValueError('One original frontend and six coherent propagations')
        if kinds.count('detector_ROI')!=6 or kinds.count('electronic_residual')!=4:
            raise ValueError('Original excluded operations still execute')
        inputs={allocation_pointer(v) for v in slot['static'].values()}
        if len(inputs)!=4 or not slot.get('prefix_owned'):raise ValueError('Owned input/prefix tensors required')
        pointers=inputs|{allocation_pointer(v) for v in slot['prefix_owned'].values()}|{allocation_pointer(slot['prediction'])}
        if seen & pointers:raise ValueError('Cross-slot input/prefix/output allocation alias')
        seen.update(pointers)
        if slot.get('score_ready') is None:raise ValueError('Initial real field fence missing')


def replay_fields(torch,slots,raw,tokenize,guard,front_stream,body_stream,*,first_field,total_fields,
                  reuse_events=False,snapshot_completed=None):
    """Queue next frontend during previous body, bounded to two live fields.

    No model is loaded here. `tokenize` returns newly computed CPU ids/mask and
    exact timing/ID evidence. All callbacks are owned by the future worker.
    Raw pixels and quality retain the original outside-window preparation scope.
    Each segment's event pair is preserved for dependency-aware projection.
    """
    validate_slots(slots)
    if not 0<=first_field<total_fields or len(raw)!=32:
        raise ValueError('Original 512-video, 32-field workload required')
    if reuse_events and snapshot_completed is None:
        raise ValueError('Event reuse requires completed timestamp snapshots before overwrite')
    outputs=[];records=[]
    pools=[[],[]];pending=[None,None];snapshot_count=0
    with torch.inference_mode():
        for field in range(first_field,total_fields):
            guard.fast()
            slot_index=(field-first_field)%2;slot=slots[slot_index]
            # CPU tokenizer runs while the preceding queued GPU field executes.
            cpu_ids,cpu_mask,token_record=tokenize(field)
            if token_record['field']!=field:raise ValueError('Tokenizer field order changed')
            pin_start=time.perf_counter()
            ids=cpu_ids.pin_memory();mask=cpu_mask.bool().pin_memory()
            pin_end=time.perf_counter()
            # Wait only before REUSING this slot, not for the immediately prior
            # field. This also keeps old pinned CPU buffers alive until safe.
            wait_start=time.perf_counter();slot['score_ready'].synchronize();wait_end=time.perf_counter()
            if reuse_events and pending[slot_index] is not None:
                previous=pending[slot_index]
                snapshot_start=time.perf_counter()
                records[previous]=snapshot_completed(records[previous])
                records[previous]['event_snapshot_before_reuse']=dict(reused_at_field=field,
                    host_start=snapshot_start,host_end=time.perf_counter(),prior_score_synchronized=True)
                snapshot_count+=1
            cursor=0
            def event():
                nonlocal cursor
                if not reuse_events:return torch.cuda.Event(enable_timing=True)
                pool=pools[slot_index]
                if cursor==len(pool):pool.append(torch.cuda.Event(enable_timing=True))
                result=pool[cursor];cursor+=1
                return result
            current=raw[field%32];observed=[]
            with torch.cuda.stream(front_stream):
                h0=event();h1=event()
                h2d_host_start=time.perf_counter()
                h0.record();slot['static']['ids'].copy_(ids,non_blocking=True)
                slot['static']['mask'].copy_(mask,non_blocking=True);h1.record()
                h2d_host_end=time.perf_counter()
                # Device-to-device refresh remains charged, just as before.
                x=event();y=event()
                submit_start=time.perf_counter();x.record()
                for name in ('pixels','quality'):slot['static'][name].copy_(current[name],non_blocking=True)
                slot['segments'][0]['graph'].replay();y.record()
                observed.append(dict(kind='frontend',label='input_refresh_and_original_prefix',start=x,end=y,stream='frontend',
                    host_submit_start=submit_start,host_submit_end=time.perf_counter()))
            with torch.cuda.stream(body_stream):
                body_stream.wait_event(y)
                for segment in slot['segments'][1:]:
                    a=event();b=event()
                    submit_start=time.perf_counter();a.record();segment['graph'].replay();b.record()
                    observed.append(dict(kind=segment['kind'],label=segment['label'],start=a,end=b,stream='body',
                        host_submit_start=submit_start,host_submit_end=time.perf_counter()))
                # Clone each score before that slot's output may be overwritten.
                a=event();b=event()
                submit_start=time.perf_counter();a.record();prediction=slot['prediction'].clone();b.record()
                observed.append(dict(kind='electronics',label='retain_current_score',start=a,end=b,stream='body',
                    host_submit_start=submit_start,host_submit_end=time.perf_counter()))
            slot['score_ready']=b;slot['keepalive']=(ids,mask)
            outputs.append(prediction)
            records.append(dict(field=field,slot=slot_index,tokenizer=token_record,
                pin_host_start=pin_start,pin_host_end=pin_end,H2D_host_submit_start=h2d_host_start,H2D_host_submit_end=h2d_host_end,
                slot_reuse_wait_host_start=wait_start,slot_reuse_wait_host_end=wait_end,
                original_H2D_events=(h0,h1),events=observed,score_ready=b,
                host_submission_end=time.perf_counter()))
            pending[slot_index]=len(records)-1
        final_sync_start=time.perf_counter();records[-1]['score_ready'].synchronize()
        end_host=time.perf_counter()
        records[-1]['final_synchronization_observation']=dict(start=final_sync_start,end=end_host)
    return dict(outputs=outputs,records=records,last_score_ready_host=end_host,
        event_reuse=dict(enabled=reuse_events,allocated_events=sum(map(len,pools)),
            completed_fields_snapshotted_before_reuse=snapshot_count,
            initialization_and_snapshot_overhead_inside_measured_window=True),
        actual_online_tokenizer_calls=total_fields-first_field,
        independent_input_buffers=2,body_stream_serial=True,
        next_field_frontend_stream=True,all_startup_costs_owned_by_caller=True,
        fixed_optics_projection_not_computed_by_summing_spans=True)
