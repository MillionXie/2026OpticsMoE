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


def build(folder):
    folder=Path(folder);read=lambda n:json.loads((folder/n).read_text(encoding='utf-8-sig'))
    audit=read('local_independent_async_audit01.json')
    if audit['status']!='passed':raise ValueError('Independent accepted audit required first')
    for name,digest in audit['raw_sha256'].items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=digest:raise ValueError('Original raw SHA changed')
    raw=read('async_raw_timing.json');trace=read('async_causal_trace.json');cost=read('cost_report.json')
    nodes=trace['nodes'];fields=raw['fields'];mapping=event_phases(raw)
    actual={n['id']:n['duration_ms'] for n in nodes}
    projected={n['id']:(OPTICAL_PASS_MS if n['kind']=='optical_simulation' else
        0. if n['kind'] in {'detector_ROI','electronic_residual','capture_excluded','H2D'} else n['duration_ms']) for n in nodes}
    a=longest_path_partition(nodes,actual,mapping,fields)
    p=longest_path_partition(nodes,projected,mapping,fields)
    if not math.isclose(p['wall_ms16'],cost['time16_ms'],rel_tol=1e-10,abs_tol=1e-6):
        raise ValueError('Physical-phase ledger differs from accepted main time')
    work=defaultdict(float)
    for n in nodes:
        if n['kind'] not in ('dispatch','sync'):
            work[phase_of(n,mapping)]+=projected[n['id']]/fields
    keys=[k for k in LABELS if k in work or k in p['critical_path_contribution_ms16']]
    rows=[dict(block=LABELS[k],key=k,executed_work_ms16=work.get(k,0.),
               additive_wall_contribution_ms16=p['critical_path_contribution_ms16'].get(k,0.)) for k in keys]
    return dict(status='passed',source_leaf=str(folder.resolve()),raw_sha256=audit['raw_sha256'],
        fields=fields,actual_software=a,fixed_optics=p,rows=rows,
        sum_only_additive_wall_column=True,work_spans_NOT_additive_wall=True,
        frontend_internal_embedding_adapter_not_individually_measured=True,
        merger_decoder_independently_measured=raw.get('detailed_electronics_phases',False),
        physical_optical_hardware_measured=False,
        scope='OEO includes charged post-detector electronics and next-field encoding, not the excluded electronic residual branch')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('leaf',type=Path);parser.add_argument('out',type=Path)
    args=parser.parse_args();result=build(args.leaf)
    if args.out.exists() or args.out.with_suffix('.csv').exists():raise FileExistsError('Preserve existing phase ledger')
    with args.out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    with args.out.with_suffix('.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['block','key','executed_work_ms16','additive_wall_contribution_ms16'])
        writer.writeheader();writer.writerows(result['rows'])
    print(json.dumps(dict(status='passed',software_ms16=result['actual_software']['wall_ms16'],
        fixed_optics_ms16=result['fixed_optics']['wall_ms16'],blocks=len(result['rows'])),ensure_ascii=False))
