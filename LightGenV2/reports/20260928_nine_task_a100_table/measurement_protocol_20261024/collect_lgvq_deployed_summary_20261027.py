"""Recheck sealed baseline workbook inputs and aggregate all three new LG runs."""
import argparse
import copy
import json
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET
from collect_energy_summary_20261026 import collect as original_collect,point,read,sha,close
from lgvq_phase_ledger_20261026 import build as phase_build
from audit_lgvq_async_point_20261026 import compare

PROFILE='A100_NO_EXTRA_WARMUP_ELECTRONIC_GRAPH_BUILD_SEPARATE_V3'
OLD_BOOK_SHA='1ef75102c139003890912df6e61bc308e4c7ccef33cd4dd1e6e1c95e0def4692'
OBSERVED_LOCAL_BOOK_SHA='387a2c09fe0efcd30e8b772ad9aa058944eb7e6922a39a5da2ae0f3821bc26e5'


def verify_current_book(book,old):
    digest=sha(book)
    if digest not in (OLD_BOOK_SHA,OBSERVED_LOCAL_BOOK_SHA):raise ValueError('Reference workbook changed after read-only observation')
    ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with ZipFile(book) as z:
        strings=[''.join(t.itertext()) for t in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',ns)] if 'xl/sharedStrings.xml' in z.namelist() else []
        rows={}
        for row in ET.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('.//s:sheetData/s:row',ns):
            values={}
            for c in row.findall('s:c',ns):
                v=c.find('s:v',ns);inline=c.find('s:is',ns)
                value=v.text if v is not None else ''.join(inline.itertext()) if inline is not None else None
                if c.get('t')=='s':value=strings[int(value)]
                values[''.join(x for x in c.get('r') if x.isalpha())]=value
            rows[int(row.get('r'))]=values
    header=next(r for r,v in rows.items() if v.get('B')=='模型' and v.get('C')=='MPS/%' and v.get('E')=='每16视频/ms')
    labels={'qwen':'Qwen','glm':'GLM','clip':'CLIP','yolo':'YOLO','kimi':'Kimi','deepseek':'DeepSeek完整'}
    expected={(labels[g['model']],str(g['cap'])):g for g in old['groups']};seen=set()
    for i in range(header+1,header+1+len(expected)):
        row=rows[i];key=(row.get('B'),row.get('C'));g=expected[key]
        if key in seen:raise ValueError('Duplicate reference model/cap')
        seen.add(key)
        for col,k in [('E','time16_ms'),('F','equivalent_frame_ms'),('G','energy16_j'),('H','energy_per_frame_j')]:close(float(row[col]),g[k])
    if seen!=set(expected) or sha(book)!=digest:raise ValueError('Reference contents changed during check')
    return dict(original_sealed_sha256=OLD_BOOK_SHA,observed_local_sha256=digest,bytes=book.stat().st_size,
        original_seal_still_matches=digest==OLD_BOOK_SHA,all_12_model_cap_time_energy_values_match_sealed_inputs=True,
        current_file_preserved_without_overwrite=True)


def collect(source,leaves):
    source=Path(source).resolve();leaves=[Path(p).resolve() for p in leaves]
    if len(leaves)!=3 or len(set(leaves))!=3:raise ValueError('Exactly three distinct fresh-process LG rounds required')
    old=read(source/'collected_results.json');manifest=read(source/'delivery_manifest.json')
    if manifest['status']!='sealed':raise ValueError('Original sealed workbook required')
    for row in manifest['files']:
        p=source/row['path']
        if p.suffix.lower()=='.xlsx':continue
        if p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:raise ValueError('Original delivery changed')
    book=next(source.glob('*.xlsx'))
    provenance=verify_current_book(book,old)
    replay=original_collect([Path(p['path']) for p in old['rounds']],Path(old['LG']['path']),
        physical_scope_ledger=old['LG']['physical_scope_ledger_path'])
    if set(replay)!=set(old):raise ValueError('Original summary schema differs')
    compare(old,replay,'original_sealed_inputs')
    data=copy.deepcopy(old);data['LG_historical']=data['LG'];data['LG_rounds']=[];phases=[]
    index={r['local_path']:r for r in data['raw_file_index']};commits=set();pids=set()
    def local_file(p,role):
        p=Path(p).resolve();index[str(p)]=dict(local_path=str(p),remote_path=None,bytes=p.stat().st_size,sha256=sha(p),role=role)
    for repeat,leaf in enumerate(leaves,1):
        measured=point(leaf,'LG_async');identity=read(leaf/'identity.json');scope=read(leaf/'a100_deployed_scope_identity.json')
        if scope['profile']!=PROFILE or scope['explicit_extra_warmup_forwards']!=0:raise ValueError('Different deployment/warm scope')
        if read(leaf/'local_native_parallel_audit01.json')['status']!='passed':raise ValueError('Independent native-parallel audit required')
        phases.append(phase_build(leaf,True,exclude_graph_construction=True));v=phases[-1]
        ledger=leaf/'local_deployed_phase_ledger_v3_01.json'
        if read(ledger)!=v:raise ValueError('New v3 ledger does not replay')
        commits.add(identity['published_commit']);pids.add(identity['pid'])
        raw=read(leaf/'async_raw_timing.json');cost=read(leaf/'cost_report.json');fields=raw['fields']
        row=dict(repeat=repeat,path=str(leaf),calls=2048,fields=fields,
            software_elapsed_s=raw['actual_async_software_wall_ms']/1000,
            previous_v2_elapsed_s=v['previous_v2_time16_ms']*fields/1000,
            graph_build_host_ms=v['excluded_electronic_graph_construction_host_work_ms16']*fields,
            projected_elapsed_s=v['fixed_optics']['wall_ms16']*fields/1000,
            component_proxy_energy_j=v['fixed_optics_component_energy']['energy_j'],
            common_effective_tokens=cost['common_effective_tokens'],native_effective_tokens=cost['native_effective_tokens'],
            software_board_energy_j=raw['raw_board_energy']['energy_j'],
            software_matched_board_average_w=measured['measured_software_matched_board_average_w'],
            raw_software_time16_ms=measured['measured_software_time16_ms'],
            projected_time16_ms=v['fixed_optics']['wall_ms16'],projected_energy16_j=v['fixed_optics_component_energy']['energy16_j'],
            explicit_extra_warmup_forwards=0,published_commit=identity['published_commit'],
            max_power_sample_age_s=raw['raw_board_energy']['max_power_sample_age_s'])
        data['LG_rounds'].append(row)
        for f in measured['files']:index[f['local_path']]=f
        for name,role in [('local_native_parallel_audit01.json','local_native_parallel_audit'),
            ('local_deployed_phase_ledger_v3_01.json','local_derived_physical_scope')]:local_file(leaf/name,role)
        root=leaf.parent;receipt=read(root/'backup_receipt_complete01.json')
        for f in receipt['verified_files']:
            target=root/f['path']
            if target.stat().st_size!=f['bytes'] or sha(target)!=f['sha256']:raise ValueError('Parent closure receipt differs')
        state=read(root/'suite_status.json')
        if state['status']!='complete' or not state['mps_released']:raise ValueError('Complete released parent required')
        local_file(root/'backup_receipt_complete01.json','closed_parent_receipt');local_file(root/'external_controller.log','exact_external_controller_log')
    if len(commits)!=1 or len(pids)!=3:raise ValueError('One implementation and three distinct workers required')
    rows=data['LG_rounds'];fields=sum(r['fields'] for r in rows);calls=sum(r['calls'] for r in rows)
    software_s=sum(r['software_elapsed_s'] for r in rows);projected_s=sum(r['projected_elapsed_s'] for r in rows)
    energy=sum(r['component_proxy_energy_j'] for r in rows);board=sum(r['software_board_energy_j'] for r in rows)
    common=sum(r['common_effective_tokens'] for r in rows);native=sum(r['native_effective_tokens'] for r in rows)
    phase=copy.deepcopy(phases[0]);phase.update(source_leaves=[str(p) for p in leaves],fields=fields,aggregation='all_three_rounds_pooled')
    phase['raw_sha256_by_leaf']={str(p):v['raw_sha256'] for p,v in zip(leaves,phases)};phase['raw_sha256']={}
    for family in ('actual_software','fixed_optics'):
        phase[family]={'wall_ms16':sum(v[family]['wall_ms16']*v['fields'] for v in phases)/fields,
            'critical_path_contribution_ms16':{k:sum(v[family]['critical_path_contribution_ms16'].get(k,0)*v['fields'] for v in phases)/fields
                for k in set().union(*(v[family]['critical_path_contribution_ms16'] for v in phases))}}
    for i,r in enumerate(phase['rows']):
        for key in ('executed_work_ms16','additive_wall_contribution_ms16'):r[key]=sum(v['rows'][i][key]*v['fields'] for v in phases)/fields
    for key in ('previous_v2_time16_ms','legacy_fixed_optics_time16_ms','excluded_electronic_graph_construction_host_work_ms16',
                'critical_path_reduction_from_graph_build_ms16'):
        phase[key]=sum(v[key]*v['fields'] for v in phases)/fields
    ce=phase['fixed_optics_component_energy']
    for key in ('energy_j','controller_j','optical_devices_j','electronic_board_j','controller_union_ms','optical_union_ms','electronic_union_ms','optical_electronic_overlap_ms'):
        ce[key]=sum(v['fixed_optics_component_energy'][key] for v in phases)
    ce['energy16_j']=energy/fields
    lg=copy.deepcopy(data['LG_historical']);lg.update(path=str(leaves[0]),paths=[str(p) for p in leaves],rounds=3,video_calls=calls,
        timing_profile=PROFILE,measured_software_time16_ms=software_s*1000/fields,
        fixed_optics_counterfactual_time16_ms=projected_s*1000/fields,fixed_optics_component_proxy_energy16_j=energy/fields,
        measured_software_board_FFT_energy16_j=board/fields,measured_software_matched_board_average_w=board/software_s,
        projected_common_tokens_per_s=common/projected_s,projected_native_tokens_per_s=native/projected_s,
        measured_software_common_tokens_per_s=common/software_s,measured_software_native_tokens_per_s=native/software_s,
        actual_stream_span_overlap_ms=sum(read(p/'cost_report.json')['measured_next_frontend_prior_body_overlap_ms'] for p in leaves),
        GPU_resource_span_totals_NOT_additive={k:sum(read(p/'cost_report.json')['actual_GPU_resource_span_totals_ms'][k] for p in leaves)
            for k in lg['GPU_resource_span_totals_NOT_additive']},component_energy_proxy=ce,
        physical_scope_version='exclude_electronic_graph_build_v3',physical_scope_ledger_path=None,
        physical_scope_ledger_sha256=None,files=[])
    data['LG']=lg;phase['projected_common_tokens_per_s']=common/projected_s;phase['projected_native_tokens_per_s']=native/projected_s
    lg.pop('legacy_fixed_optics_v1',None)
    data['comparisons']=[dict(platform=g['platform'],model=g['model'],cap=g['cap'],
        baseline_over_actual_LG_software_time=g['time16_ms']/lg['measured_software_time16_ms'],
        baseline_over_projected_LG_time=g['time16_ms']/lg['fixed_optics_counterfactual_time16_ms'],
        baseline_proxy_over_LG_component_proxy_energy=g['energy16_j']/lg['fixed_optics_component_proxy_energy16_j'],
        projected_common_throughput_gain=lg['projected_common_tokens_per_s']/g['pooled_common_tokens_per_s'],
        baseline_over_LG_system_proxy_average_power=(g['energy16_j']/g['time16_ms'])/(energy/projected_s/1000),
        platform_relation='same_platform' if g['platform']==lg['platform'] else 'cross_configuration_not_controlled',
        physical_optical_speedup_measured=False,energy_proxy_models_differ=True) for g in data['groups']]
    for c in data['comparisons']:close(c['projected_common_throughput_gain'],c['baseline_over_projected_LG_time'])
    for p in (source/'delivery_manifest.json',source/'collected_results.json'):local_file(p,'original_sealed_delivery_source')
    local_file(book,'reference_workbook_verified_numeric_snapshot')
    data['raw_file_index']=sorted(index.values(),key=lambda r:r['local_path'])
    data.update(base_delivery=str(source),base_workbook_provenance=provenance,LG_deployed_profile=PROFILE,LG_round_count=3,
        LG_projection_is_not_physical_hardware_measurement=True)
    return data,phase


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('out',type=Path);p.add_argument('phase',type=Path)
    p.add_argument('leaves',nargs=3,type=Path);a=p.parse_args()
    if a.out.exists() or a.phase.exists():raise FileExistsError('Preserve immutable inputs')
    data,phase=collect(a.source,a.leaves)
    for path,value in ((a.out,data),(a.phase,phase)):
        with path.open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=True,indent=2)
    print(json.dumps(dict(status='passed',LG_rounds=3,calls=data['LG']['video_calls'],time16_ms=data['LG']['fixed_optics_counterfactual_time16_ms']),ensure_ascii=True))
