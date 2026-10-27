"""Data-only physical-phase ledger. Work spans and wall contributions differ.

Read an already qualified, backed async leaf; never overwrite raw or audits.
The additive column is one witnessed longest path, not a sum of overlapping
streams. The work column reports all phase executions, including hidden work.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

from lgvq_async_trace_contract_20261026 import critical_path, OPTICAL_PASS_MS

PASSES=('parallel_router.forward','parallel_optics.expert','parallel_optics.global_path',
        'serial_router.forward','serial_optics.expert','serial_optics.global_path')
LABELS={
    'tokenizer':'文字tokenizer', 'frontend':'输入前处理（embedding、位置、adapter合并）',
    'vision_entry':'视觉光路入口编码', 'merger':'merger与序列组装',
    'language_entry':'语言光路入口编码', 'decoder':'decoder与最终评分',
    'startup':'首次真实建图准备', 'scheduling':'调度、CPU返回与同步',
    'excluded':'FFT仿真、ROI、电残差及原H2D（光电口径不计）'}
for i in range(1,7):
    LABELS[f'optics{i}']=f'{"vision" if i<=3 else "language"}光传播{i if i<=3 else i-3}'
    LABELS[f'oeo{i}']=f'OEO{i}（路由/读出、融合及下一段编码）'


def event_phases(raw):
    result={};detail=raw.get('detailed_electronics_phases',False)
    for row in raw['records']:
        field=row['field'];prefix=f'f{field}';active=0;current='vision_entry';seen=[];details=set()
        body_sequence=0
        for i,e in enumerate(row['events']):
            if e['stream']=='frontend':
                result[f'{prefix}.g{i}']='frontend';continue
            # An eager gap precedes this event, so belongs to the previous block.
            result[f'{prefix}.gap{body_sequence}']=current
            kind=e['kind'];label=e['label']
            if kind=='optical_simulation':
                if label!=PASSES[len(seen)]:raise ValueError('Original six optical pass order changed')
                seen.append(label);active=len(seen);phase=f'optics{active}';current=f'oeo{active}'
            elif kind in ('detector_ROI','electronic_residual'):
                phase='excluded'
            elif label=='frame_merger' or label=='after_frame_merger':
                details.add('frame_merger');phase='merger';current='merger'
            elif label in ('readout','after_readout','retain_current_score'):
                if label in ('readout','after_readout'):details.add('readout')
                phase='decoder';current='decoder'
            elif active==3 and label=='after_language_routes.0' and detail:
                phase='language_entry';current=phase
            else:
                phase=current
            result[f'{prefix}.g{i}']=phase;body_sequence+=1
        if tuple(seen)!=PASSES:raise ValueError('Six full original coherent passes required')
        if detail and details!={'frame_merger','readout'}:raise ValueError('Independent merger/decoder events missing')
        result[prefix+'.tail']='decoder' if detail else current
    return result


def phase_of(node,mapping):
    kind=node['kind']
    if kind in ('dispatch','sync'):return 'scheduling'
    if kind=='tokenizer':return 'tokenizer'
    if kind=='capture_electronics':return 'startup'
    if kind in ('capture_excluded','H2D','detector_ROI','electronic_residual'):return 'excluded'
    if kind=='frontend':return 'frontend'
    return mapping.get(node['id'],'scheduling')


def longest_path_partition(nodes,durations,mapping,fields):
    schedule=critical_path(nodes,durations);by={n['id']:n for n in nodes}
    key=max(schedule['ends_ms'],key=schedule['ends_ms'].get);chain=[]
    while True:
        n=by[key];chain.append(n)
        if not n['parents']:break
        key=max(n['parents'],key=lambda p:(schedule['ends_ms'][p],p))
    phases=defaultdict(float)
    for n in chain:phases[phase_of(n,mapping)]+=durations[n['id']]/fields
    total=schedule['wall_ms']/fields
    if not math.isclose(sum(phases.values()),total,rel_tol=1e-10,abs_tol=1e-6):
        raise ValueError('Phase longest-path partition does not close to wall')
    return dict(wall_ms16=total,critical_path_contribution_ms16=dict(phases),
                critical_path_node_ids=[n['id'] for n in reversed(chain)])


def software_only_submissions(raw):
    """Only precisely timed software-only calls, never generic queue gaps."""
    excluded={'optical_simulation','detector_ROI','electronic_residual'}
    ids=set()
    for row in raw['records']:
        prefix=f"f{row['field']}"
        ids.add(prefix+'.H2Dsubmit')
        for i,event in enumerate(row['events']):
            if event['kind'] in excluded:ids.add(prefix+f'.submit{i}')
    return ids


def electronic_graph_build_nodes(raw,nodes):
    if raw.get('explicit_warmup_forwards')!=0:raise ValueError('No extra warmup permitted')
    expected={f"f{row['field']}.capture{i}" for row in raw['records']
        for i,segment in enumerate(row.get('graph_construction',[]))
        if segment['kind'] not in {'optical_simulation','detector_ROI','electronic_residual'}}
    selected={n['id'] for n in nodes if n['kind']=='capture_electronics'}
    if (selected!=expected or len(selected)!=44
            or {n['field'] for n in nodes if n['id'] in selected}!={3,4}):
        raise ValueError('Exactly 44 witnessed electronic graph constructions required')
    return selected


def build(folder,exclude_software_submit=False,*,warmed_diagnostic=False,exclude_graph_construction=False):
    folder=Path(folder);read=lambda n:json.loads((folder/n).read_text(encoding='utf-8-sig'))
    audit=read('local_warmed_async_audit01.json' if warmed_diagnostic else 'local_independent_async_audit01.json')
    if audit['status']!='passed':raise ValueError('Independent accepted audit required first')
    if warmed_diagnostic and (audit.get('timing_profile')!='ONE_EXTRA_REAL_WARMUP_FIELD_DIAGNOSTIC'
            or audit.get('original_cold_scope_qualified') is not False):
        raise ValueError('Explicit separate warmed-diagnostic audit required')
    for name,digest in audit['raw_sha256'].items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=digest:raise ValueError('Original raw SHA changed')
    raw=read('async_raw_timing.json');trace=read('async_causal_trace.json');cost=read('cost_report.json')
    nodes=trace['nodes'];fields=raw['fields'];mapping=event_phases(raw)
    actual={n['id']:n['duration_ms'] for n in nodes}
    projected={n['id']:(OPTICAL_PASS_MS if n['kind']=='optical_simulation' else
        0. if n['kind'] in {'detector_ROI','electronic_residual','capture_excluded','H2D'} else n['duration_ms']) for n in nodes}
    a=longest_path_partition(nodes,actual,mapping,fields)
    legacy=longest_path_partition(nodes,projected,mapping,fields)
    if not math.isclose(legacy['wall_ms16'],cost['time16_ms'],rel_tol=1e-10,abs_tol=1e-6):
        raise ValueError('Physical-phase ledger differs from accepted main time')
    submission_ids=software_only_submissions(raw) if exclude_software_submit else set()
    by_id={n['id']:n for n in nodes}
    if not submission_ids<=set(by_id) or any(by_id[k]['kind']!='dispatch' for k in submission_ids):
        raise ValueError('Explicit software-only CPU submission evidence missing')
    for key in submission_ids:projected[key]=0.
    prior_v2=longest_path_partition(nodes,projected,mapping,fields)
    construction_ids=set()
    if exclude_graph_construction:
        if warmed_diagnostic or not exclude_software_submit:raise ValueError('Only original no-warmup v2 may become deployed v3')
        scope=read('a100_deployed_scope_identity.json')
        if (scope.get('profile')!='A100_NO_EXTRA_WARMUP_ELECTRONIC_GRAPH_BUILD_SEPARATE_V3'
                or scope.get('explicit_extra_warmup_forwards')!=0 or scope.get('all_other_v2_nodes_retained') is not True):
            raise ValueError('Predeclared deployed scope required')
        construction_ids=electronic_graph_build_nodes(raw,nodes)
        for key in construction_ids:projected[key]=0.
    p=longest_path_partition(nodes,projected,mapping,fields)
    from lgvq_async_component_energy_20261026 import component_energy
    energy=component_energy(nodes,critical_path(nodes,projected),fields)
    work=defaultdict(float)
    for n in nodes:
        work[phase_of(n,mapping)]+=projected[n['id']]/fields
    order=['tokenizer','frontend','vision_entry']
    for i in (1,2,3):order.extend((f'optics{i}',f'oeo{i}'))
    order.extend(('merger','language_entry'))
    for i in (4,5,6):order.extend((f'optics{i}',f'oeo{i}'))
    order.extend(('decoder','startup','scheduling','excluded'))
    keys=[k for k in order if k in work or k in p['critical_path_contribution_ms16']]
    rows=[dict(block=LABELS[k],key=k,executed_work_ms16=work.get(k,0.),
               additive_wall_contribution_ms16=p['critical_path_contribution_ms16'].get(k,0.)) for k in keys]
    reference=None
    ref_folder=folder.parent.parent/'a100_lgvq_async_20261026_01'/'lg_repeat01'
    if exclude_software_submit and ref_folder.resolve()!=folder.resolve() and ref_folder.exists():
        ref=build(ref_folder,True)
        reference=dict(source_leaf=ref['source_leaf'],raw_sha256=ref['raw_sha256'],
            same_scope_time16_ms=ref['fixed_optics']['wall_ms16'],
            legacy_scope_time16_ms=ref['legacy_fixed_optics_time16_ms'])
    result=dict(status='passed',source_leaf=str(folder.resolve()),raw_sha256=audit['raw_sha256'],
        fields=fields,actual_software=a,fixed_optics=p,rows=rows,
        scope_version='exclude_software_CPU_submit_v2' if exclude_software_submit else 'original_CPU_dispatch_retained_v1',
        legacy_fixed_optics_time16_ms=legacy['wall_ms16'],
        software_only_CPU_submission_node_ids=sorted(submission_ids),
        software_only_CPU_submission_work_ms16=sum(by_id[k]['duration_ms'] for k in submission_ids)/fields,
        observed_queue_latencies_retained_not_assumed_removed=True,
        reference_round=reference,
        fixed_pass_components_ms=dict(SLM=1000/1400,CCD_exposure=.300,camera_readout=.0307),
        fixed_optics_component_energy=energy,
        projected_common_tokens_per_s=cost['common_effective_tokens']/(p['wall_ms16']*fields/1000),
        projected_native_tokens_per_s=cost['native_effective_tokens']/(p['wall_ms16']*fields/1000),
        sum_only_additive_wall_column=True,work_spans_NOT_additive_wall=True,
        scheduling_span_includes_CPU_actions_and_observed_queue_not_kernel_work=True,
        frontend_internal_embedding_adapter_not_individually_measured=True,
        merger_decoder_independently_measured=raw.get('detailed_electronics_phases',False),
        physical_optical_hardware_measured=False,
        scope='OEO includes charged post-detector electronics and next-segment encoding, not the excluded electronic residual branch')
    if warmed_diagnostic:
        result.update(timing_profile=audit['timing_profile'],original_cold_scope_qualified=False,
            qualified_for_final_hybrid_table=False,extra_pre_window_real_fields=1,
            scope_version='WARMED_DIAGNOSTIC_'+result['scope_version'],warmup_accounting=audit['warmup_accounting'],
            fixed_optics_result_is_measured_main_window_only=True,
            warmup_inclusive_optical_projection_not_claimed=True,Qwen_baseline_also_warmed=False)
    if exclude_graph_construction:
        result.update(timing_profile=scope['profile'],scope_version='exclude_electronic_graph_build_v3',
            original_cold_scope_qualified=False,qualified_for_new_deployed_scope=True,
            extra_pre_window_real_fields=0,previous_v2_time16_ms=prior_v2['wall_ms16'],
            excluded_electronic_graph_construction_node_ids=sorted(construction_ids),
            excluded_electronic_graph_construction_host_work_ms16=sum(by_id[k]['duration_ms'] for k in construction_ids)/fields,
            critical_path_reduction_from_graph_build_ms16=prior_v2['wall_ms16']-p['wall_ms16'],
            baseline_scope_unchanged=True,software_wall_still_includes_all_graph_construction=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('leaf',type=Path);parser.add_argument('out',type=Path)
    parser.add_argument('--exclude-software-submit',action='store_true')
    parser.add_argument('--warmed-diagnostic',action='store_true',help='Separate profile, not original cold table')
    parser.add_argument('--exclude-electronic-graph-construction',action='store_true')
    args=parser.parse_args();result=build(args.leaf,args.exclude_software_submit,warmed_diagnostic=args.warmed_diagnostic,
        exclude_graph_construction=args.exclude_electronic_graph_construction)
    if args.out.exists() or args.out.with_suffix('.csv').exists():raise FileExistsError('Preserve existing phase ledger')
    with args.out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    with args.out.with_suffix('.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['block','key','executed_work_ms16','additive_wall_contribution_ms16'])
        writer.writeheader();writer.writerows(result['rows'])
    print(json.dumps(dict(status='passed',software_ms16=result['actual_software']['wall_ms16'],
        fixed_optics_ms16=result['fixed_optics']['wall_ms16'],blocks=len(result['rows'])),ensure_ascii=False))
