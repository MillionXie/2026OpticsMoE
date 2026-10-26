"""Original async worker events -> witnessed host/stream/buffer causal DAG.

No wall residual is appended. GPU clock origin must lie inside the measured
origin bracket AND every observed CPU-submit/GPU-sync causality bound. CPU waits
are GPU dependencies plus measured return overhead, never frozen blocking time.
Queue latency is an explicit measured ready-to-start interval, held fixed only
for the labelled counterfactual. Eager electronic envelopes retain dispatch and
kernel spans; they are not claimed to be kernel-only operator profiles.
"""
import math

from lgvq_async_observed_dag_20261026 import measured_dag
from lgvq_async_trace_contract_20261026 import project
from lgvq_async_component_energy_20261026 import interval_union, length


def prepare_records(raw):
    rows = raw['records']
    if (len(rows) != 128 or [r['field'] for r in rows] != list(range(128))
            or raw['video_calls'] != 2048 or raw['unique_videos'] != 512 or raw['cycles'] != 4
            or raw.get('all_predictions_bitwise_equal') is not True
            or raw.get('startup_and_both_actual_captures_included') is not True):
        raise ValueError('Complete original workload, scoring and startup evidence required')
    age=raw['raw_board_energy']['max_power_sample_age_s']
    if not isinstance(age,(int,float)) or not math.isfinite(age) or not 0<=age<=.25:
        raise ValueError('Original 250-ms telemetry acceptance is unchanged')
    start = raw['host_start']
    ms = lambda value: (value-start)*1000
    actions = []; ops = []; scores = {}; fronts = {}; transfers = {}
    def action(name, kind, a, b, field, *, wait_for=()):
        item = dict(id=name, kind=kind, a=ms(a), b=ms(b), field=field,
                    wait_for=list(wait_for), witness=name)
        actions.append(item)
        return name
    def op(name, kind, a, b, stream, submit, field, *, opaque=False):
        item = dict(id=name, kind=kind, cuda_a=a, cuda_b=b, stream=stream,
                    submit=submit, field=field, opaque=opaque, extra_parents=[])
        ops.append(item)
        return item
    for row in rows:
        field = row['field']; prefix = f'f{field}'; token = row['tokenizer']
        if token['field'] != field:
            raise ValueError('Original tokenizer field order differs')
        if (len(token['input_ids'])!=1 or len(token['attention_mask'])!=1
                or not token['input_ids'][0] or len(token['input_ids'][0])!=len(token['attention_mask'][0])):
            raise ValueError('Actual per-field shared prompt IDs/masks required')
        action(prefix+'.token', 'tokenizer', token['start'], token['end'], field)
        if 'pin_host_start' in row:
            action(prefix+'.pin', 'dispatch', row['pin_host_start'], row['pin_host_end'], field)
        h = action(prefix+'.H2Dsubmit', 'dispatch', row['H2D_host_submit_start'], row['H2D_host_submit_end'], field)
        a,b = row['original_H2D_cuda_ms']
        op(prefix+'.H2D', 'H2D', a,b, 'frontend', h,field)
        transfers[field] = prefix+'.H2D'
        front_events = [(i,e) for i,e in enumerate(row['events']) if e['stream']=='frontend']
        if len(front_events)!=(2 if field in (3,4) else 1) or any(e['kind']!='frontend' for _,e in front_events):
            raise ValueError('Actual frontend evidence missing')
        for i,e in front_events:
            host = action(prefix+f'.submit{i}', 'dispatch', e['host_submit_start'],e['host_submit_end'],field)
            o = op(prefix+f'.g{i}',e['kind'],e['cuda_start_ms'],e['cuda_end_ms'],'frontend',host,field)
            fronts[field] = o['id']
        body_events = [(i,e) for i,e in enumerate(row['events']) if e['stream']=='body']
        body_kinds=[e['kind'] for _,e in body_events]
        if (body_kinds.count('optical_simulation')!=6 or body_kinds.count('detector_ROI')!=6
                or body_kinds.count('electronic_residual')!=4
                or body_kinds.count('electronics')!=(0 if field<3 else 18)):
            raise ValueError('All original simulation/ROI/residual/electronic segments must execute')
        if field < 3:
            if row.get('mode') != 'eager_real' or len(body_events) != 16:
                raise ValueError('The first three real eager fields must be retained')
            a = row['body_envelope_cuda_ms'][0]
            host_a = row['body_host_start']
            for sequence,(i,e) in enumerate(body_events):
                host = action(prefix+f'.eagergap{sequence}', 'dispatch',host_a,e['host_submit_start'],field)
                op(prefix+f'.gap{sequence}','electronics',a,e['cuda_start_ms'],'body',host,field,opaque=True)
                host = action(prefix+f'.submit{i}','dispatch',e['host_submit_start'],e['host_submit_end'],field)
                op(prefix+f'.g{i}',e['kind'],e['cuda_start_ms'],e['cuda_end_ms'],'body',host,field)
                a = e['cuda_end_ms']; host_a = e['host_submit_end']
            host = action(prefix+'.eagertail','dispatch',host_a,row['body_host_end'],field)
            o=op(prefix+'.tail','electronics',a,row['body_envelope_cuda_ms'][1],'body',host,field,opaque=True)
        else:
            if field in (3,4) and row.get('mode') != 'capture_real':
                raise ValueError('Both independent real capture fields required')
            for i,e in body_events:
                host=action(prefix+f'.submit{i}','dispatch',e['host_submit_start'],e['host_submit_end'],field)
                o=op(prefix+f'.g{i}',e['kind'],e['cuda_start_ms'],e['cuda_end_ms'],'body',host,field)
            if not body_events:
                raise ValueError('Original complete body trace missing')
        scores[field] = o['id']
        if not math.isclose(o['cuda_b'],row['score_ready_cuda_ms'],rel_tol=0,abs_tol=.001):
            raise ValueError('Score-ready terminal event does not match original body')
        for j,s in enumerate(row.get('graph_construction',[])):
            kind = 'capture_excluded' if s['kind'] in {'optical_simulation','detector_ROI','electronic_residual'} else 'capture_electronics'
            action(prefix+f'.capture{j}',kind,s['capture_host_start'],s['capture_host_end'],field)
        if field in (3,4) and len(row.get('graph_construction',[])) != 34:
            raise ValueError('Original 34 independent capture segments missing')
        if field>=5 and 'slot_reuse_wait_host_start' not in row:
            raise ValueError('Original i-2 slot reuse fence missing')
        if 'slot_reuse_wait_host_start' in row:
            if field < 5 or row['slot'] != (field-5)%2:
                raise ValueError('Original two-slot field reuse changed')
            action(prefix+'.slotwait','sync',row['slot_reuse_wait_host_start'],row['slot_reuse_wait_host_end'],field,
                   wait_for=[scores[field-2]])
        for j,s in enumerate(row.get('synchronization_observations',[])):
            name=s['kind']
            if name=='device_before_buffers':
                deps=[scores[field-1]]
            elif name=='device_after_transfer':
                deps=[prefix+f'.g{front_events[0][0]}',transfers[field]]
            elif name=='frontend_before_body_capture':
                deps=[fronts[field]]
            elif name=='score_ready':
                deps=[scores[field]]
            else:
                raise ValueError('Unknown original synchronization boundary')
            action(prefix+f'.sync{j}','sync',s['start'],s['end'],field,wait_for=deps)
        sync_kinds={s['kind'] for s in row.get('synchronization_observations',[])}
        expected={'score_ready'} if field<3 else (
            {'device_before_buffers','device_after_transfer','frontend_before_body_capture','score_ready'} if field<5 else set())
        if sync_kinds!=expected:
            raise ValueError('Original real-field synchronization evidence missing')
        if 'final_synchronization_observation' in row:
            s=row['final_synchronization_observation']
            action(prefix+'.finalsync','sync',s['start'],s['end'],field,wait_for=[scores[field]])
    by_action={a['id']:a for a in actions}
    for field in range(128):
        body=[o for o in ops if o['field']==field and o['stream']=='body']
        body[0]['extra_parents'].append(fronts[field])
    if rows[-1].get('final_synchronization_observation',{}).get('end') != raw['host_end']:
        raise ValueError('Missing actual terminal GPU-score synchronization return')
    return actions,ops,by_action,scores


def clock_origin(raw, actions, ops):
    """Feasible CUDA-origin offset, in ms relative to measured host_start."""
    start=raw['host_start'];lo,hi=[(x-start)*1000 for x in raw['origin_host_clock_bracket']]
    by_action={a['id']:a for a in actions};by_op={o['id']:o for o in ops}
    if lo>hi:
        raise ValueError('Invalid original clock bracket')
    for o in ops:
        if not o['opaque']:
            lo=max(lo,by_action[o['submit']]['a']-o['cuda_a'])
    for action in actions:
        for parent in action['wait_for']:
            hi=min(hi,action['b']-by_op[parent]['cuda_b'])
    if lo>hi:
        raise ValueError('No causal host/CUDA clock alignment; preserve raw, do not shift clocks arbitrarily')
    return (lo+hi)/2,dict(feasible_origin_ms=[lo,hi],chosen_origin_ms=(lo+hi)/2,
                        all_submit_and_sync_bounds_satisfied=True,clock_values_modified=False)


def observations(raw):
    actions,ops,by_action,scores=prepare_records(raw)
    offset,clock=clock_origin(raw,actions,ops)
    by_op={o['id']:o for o in ops};nodes=[];ends={}
    for o in ops:
        o['a']=o['cuda_a']+offset;o['b']=o['cuda_b']+offset
        if o['opaque']:
            o['a']=max(o['a'],by_action[o['submit']]['a'])
        if o['a']>o['b']:
            raise ValueError('Eager electronic envelope violates its actual CPU submission bound')
        ends[o['id']]=o['b']
    def node(name,kind,a,b,parents,resource,witness,field=None):
        if a<0 or b<a:
            raise ValueError('Original measured interval is invalid: '+name)
        nodes.append(dict(id=name,kind=kind,observed_start_ms=a,observed_end_ms=b,
                          parents=list(dict.fromkeys(parents)),resource=resource,witness=witness,field=field))
        ends[name]=b
    node('origin','dispatch',0.,0.,[],None,'actual host_start')
    previous='origin';previous_end=0.
    actions.sort(key=lambda a:(a['a'],a['b'],a['id']))
    for action in actions:
        a,b=action['a'],action['b'];name=action['id'];field=action['field']
        if a<previous_end-1e-6:
            raise ValueError('Overlapping original CPU actions; a nested/blocked interval is unaccounted')
        if a>previous_end:
            gap=name+'.hostgap'
            node(gap,'dispatch',previous_end,a,[previous],'host',dict(boundaries=[previous,name],scope='observed_between_CPU_actions'),field)
            previous=gap
        begin=name+'.begin'
        node(begin,'dispatch',a,a,[previous],None,action['witness'],field)
        parents=[begin]+action['wait_for']
        ready=max([a]+[ends[p] for p in action['wait_for']])
        node(name,action['kind'],ready,b,parents,'host',dict(original_call=[a,b],boundary=action['witness'],
             GPU_blocking_is_dependency_not_fixed_duration=bool(action['wait_for'])),field)
        previous=name;previous_end=b
    if not math.isclose(previous_end,raw['actual_async_software_wall_ms'],rel_tol=0,abs_tol=1e-5):
        raise ValueError('CPU trace has no actual terminal observation; never pad a wall residual')
    # Each CUDA stream is a serial resource. Front and body streams can overlap.
    for stream in ('frontend','body'):
        previous=None
        stream_ops=sorted((o for o in ops if o['stream']==stream),key=lambda o:(o['cuda_a'],o['cuda_b']))
        for o in stream_ops:
            parents=[o['submit']+'.begin']+o['extra_parents']+([previous] if previous else [])
            ready=max(ends[p] for p in parents)
            if o['a']<ready-1e-6:
                raise ValueError('Original GPU event precedes a required stream/CPU/buffer dependency: '+o['id'])
            if o['a']>ready:
                wait=o['id']+'.queue'
                node(wait,'dispatch',ready,o['a'],parents,stream,
                     dict(scope='observed_ready_to_GPU_start_latency_held_fixed_in_counterfactual',
                          original_CUDA_start=o['cuda_a'],aligned_ready=ready),o['field'])
                parents=[wait]
            node(o['id'],o['kind'],o['a'],o['b'],parents,stream,
                 dict(original_CUDA=[o['cuda_a'],o['cuda_b']],host_submit=o['submit'],
                      eager_interval_is_electronics_and_dispatch_not_kernel_only=o['opaque']),o['field'])
            previous=o['id']
    maximum=max(o['cuda_b'] for o in ops)
    tolerance=max(1e-4,maximum*2**-22)
    checked=measured_dag(nodes,measured_wall_ms=raw['actual_async_software_wall_ms'],clock_tolerance_ms=tolerance)
    return checked,clock,ops


def adapt(raw):
    checked,clock,ops=observations(raw)
    trace=dict(status='complete',all_predictions_bitwise_equal=True,fields=128,video_calls=2048,
               all_startup_and_capture_retained=True,new_inputs_each_field=True,nodes=checked['nodes'],
               measured_software_wall_ms=raw['actual_async_software_wall_ms'])
    projection=project(trace)
    # Actual overlap is measured using the common CUDA event clock; it is not
    # inferred from a projected speedup or a sum of individual durations.
    overlap=[]
    for field in range(1,128):
        front=[o for o in ops if o['field']==field and o['stream']=='frontend' and o['kind']!='H2D']
        prior=[o for o in ops if o['field']==field-1 and o['stream']=='body']
        overlap += [(max(a['a'],b['a']),min(a['b'],b['b'])) for a in front for b in prior
                    if min(a['b'],b['b'])>max(a['a'],b['a'])]
    stage_totals={kind:sum(o['b']-o['a'] for o in ops if o['kind']==kind)
                  for kind in {o['kind'] for o in ops}}
    native=sum(sum(bool(x) for x in r['tokenizer']['attention_mask'][0]) for r in raw['records'])
    result=dict(status='complete',system='LG_async_fixed_optics_counterfactual',video_calls=2048,
                fields=128,unique_videos=512,cycles=4,actual_async_software_wall_ms=raw['actual_async_software_wall_ms'],
                time16_ms=projection['projected_time16_ms'],energy16_j=projection['phase_energy_proxy']['energy16_j'],
                common_effective_tokens=913408,common_tokens_per_s=913408/(projection['projected_wall_ms']/1000),
                native_effective_tokens=native,native_tokens_per_s=native/(projection['projected_wall_ms']/1000),
                native_token_scope='Actual shared prompt ids once per 16-video optical field; not 16 duplicated prompts',
                phase_energy_proxy=projection['phase_energy_proxy'],fixed_optics_projection=projection,
                measured_next_frontend_prior_body_overlap_ms=length(overlap),
                measured_overlap_intervals_ms=interval_union(overlap),
                overlap_evidence='CUDA event spans on separate streams, not per-kernel active occupancy',
                kernel_occupancy_concurrency_proven=False,
                actual_GPU_resource_span_totals_ms=stage_totals,
                stage_totals_are_NOT_additive_wall_or_energy_decomposition=True,
                causal_clock_alignment=clock,every_observed_node_start_reproduced=True,
                all_predictions_bitwise_equal=True,reference_sha256=raw['reference_sha256'],
                raw_board_energy=raw['raw_board_energy'],raw_FFT_board_energy_not_used_as_optical_proxy=True,
                physical_hardware_time_measured=False,qualified_for_final_hybrid_table=True,
                arbitrary_wall_padding=False,trace_nodes=len(trace['nodes']),
                changed_optical_timing_may_change_GPU_contention=True)
    return trace,result
