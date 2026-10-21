"""CPU replay of prepared-input model execution and recorded internal gaps.

Raw reports remain immutable. No timing guesses or logging subtraction. Qwen's
approved historical100 anchor is the explicit, disclosed accounting exception.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,statistics,subprocess
from copy import deepcopy
from datetime import datetime,timezone,timedelta
from pathlib import Path
import numpy as np
from . import audit_mps_batch_activity as audit
from .. import measure_mps_batch_activity as run

REPO=run.REPO
BASE=run.ROOT
PROTOCOL='prepared_input_model_call_through_cpu_return_plus_internal_gaps_v2'
_verified=set()

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def rows(p):return list(csv.DictReader(p.open(newline='',encoding='utf-8-sig')))
def close(a,b):assert math.isclose(float(a),float(b),abs_tol=1e-6,rel_tol=1e-10),(a,b)


def replay(folder):
    folder=Path(folder).resolve()
    report=read(folder/'report.json');identity=read(folder/'identity.json')
    status=read(folder/'status.json')
    assert report['status']==status['status']=='complete' and status['samples']==558
    assert report['explicit_warmup_forwards']==0 and report['test_videos']==558 and report['cost_videos']==200
    assert report['manifest_sha256']==run.original.frozen.MANIFEST_SHA
    assert identity['checkpoint_sha256']==report['checkpoint_sha256']
    assert identity['assets']==report['assets']
    assert report['visible_multiprocessors']==(108 if report['cap']==100 else 54)
    key=(report['git_commit'],tuple(sorted(report['sources'].items())))
    if key not in _verified:
        for name,digest in report['sources'].items():
            blob=subprocess.check_output(['git','show',key[0]+':'+name.replace('\\','/')],cwd=REPO)
            assert hashlib.sha256(blob).hexdigest()==digest,(name,key[0])
        _verified.add(key)
    json_calls=(folder/'calls.json').exists()
    calls=read(folder/'calls.json') if json_calls else rows(folder/'calls.csv')
    charged=[c for c in calls if c['in_cost_subset'] in (True,'True')]
    predictions=rows(folder/'predictions.csv')
    sample_ids=([sid for c in calls for sid in c['sample_ids']] if json_calls else [c['sample_id'] for c in calls])
    manifest=run.TASK/'runs/simulation/lgvq_temporal_reproduction_a100_20261013/lgvq_train2250_test558.csv'
    assert sha(manifest)==run.original.frozen.MANIFEST_SHA
    expected=[r for r in rows(manifest) if r['split']=='test']
    assert sample_ids==[r['sample_id'] for r in predictions]==[r['sample_id'] for r in expected]
    assert len(sample_ids)==558 and len(set(sample_ids[:200]))==200
    assert [float(r['target_mos']) for r in predictions]==[float(r['temporal']) for r in expected]
    assert sum(int(c.get('videos',1)) for c in charged)==200
    if json_calls:
        assert [(c['sample_index'],c['sample_index']+c['videos']) for c in calls]==run.batches(report['batch'])
    else:assert len(charged)==200
    for c in calls:
        p,a,s,e,r=(float(c[k]) for k in ('preprocess_start','model_call_start','first_block_start','gpu_output','call_return'))
        assert p<=a<=s<e<=r
        close((e-s)*1000,c['legacy_host_ms'])
    for event in read(folder/'process_audit.json'):
        allowed={str(identity['pid'])}|{str(p) for p in event['servers']}
        assert all(str(p[0]).strip() in allowed and str(p[1]).strip()==identity['gpu_uuid'] for p in event['processes'])
    raw=rows(folder/'telemetry.csv')
    stamps=np.array([float(r['host_monotonic_s']) for r in raw])
    watts=np.array([float(r['watts']) for r in raw])
    assert np.all(np.diff(stamps)>0) and np.all(np.isfinite(watts)) and np.all(watts>0)
    intervals,model_s,gap_s=run.ready_input_intervals(charged)
    # Independently form the included pieces, without the producer's builder.
    independently=[(float(c['model_call_start']),float(c['call_return'])) for c in charged]
    independently += [(float(a['call_return']),float(b['preprocess_start'])) for a,b in zip(charged,charged[1:])]
    independently=sorted((a,b) for a,b in independently if a<b)
    assert [(a,b) for a,b in intervals if a<b]==independently
    seconds=sum(b-a for a,b in independently)
    # Second time identity: total observed wall span minus all prepare/H2D spans.
    elapsed=(float(charged[-1]['call_return'])-float(charged[0]['preprocess_start'])
        -sum(float(c['model_call_start'])-float(c['preprocess_start']) for c in charged))
    close(seconds,elapsed);close(seconds,model_s+gap_s)
    def integral(spans):
        assert stamps[0]<=min(a for a,b in spans)
        right=np.r_[stamps[1:],max(stamps[-1],max(b for a,b in spans))]
        for a,b in spans:
            used=np.flatnonzero(np.maximum(0,np.minimum(right,b)-np.maximum(stamps,a)))
            assert len(used) and max(min(right[i],b)-stamps[i] for i in used)<=.25
        return sum(float(np.dot(np.maximum(0,np.minimum(right,b)-np.maximum(stamps,a)),watts)) for a,b in spans)
    gpu=integral(independently)
    samples=[run.original.telemetry.Sample(host_time_unix=float(r['host_time_unix']),
        host_monotonic_s=float(r['host_monotonic_s']),phase=r['phase'],watts=float(r['watts']),
        utilization_percent=float(r['utilization_percent']),memory_used_mib=float(r['memory_used_mib'])) for r in raw]
    result=run.original.energy.interval_energy([(a,b) for a,b in intervals if a<b],samples,338.2)
    close(result['elapsed_s'],seconds);close(result['energy_j'],gpu+338.2*seconds)
    legacy=[]
    for i,c in enumerate(charged):
        if i:legacy.append((float(charged[i-1]['gpu_output']),float(c['preprocess_start'])))
        legacy.append((float(c['first_block_start']),float(c['gpu_output'])))
    legacy_s=sum(b-a for a,b in legacy)
    close(legacy_s*80,report['cost']['time16_ms'])
    close((integral(legacy)+338.2*legacy_s)*.08,report['cost']['energy16_j'])
    front_s=sum(float(c['first_block_start'])-float(c['model_call_start']) for c in charged)
    tail_s=float(charged[-1]['call_return'])-float(charged[-1]['gpu_output'])
    close(seconds,legacy_s+front_s+tail_s)
    actual=audit.metrics(np.array([float(r['target_mos']) for r in predictions]),np.array([float(r['prediction']) for r in predictions]))
    close(actual['srcc'],report['performance']['srcc']);close(actual['plcc'],report['performance']['plcc'])
    if 'ready_input_cost' in report:
        close(report['ready_input_cost']['time16_ms'],seconds*80)
        close(report['ready_input_cost']['energy16_j'],result['energy_j']*.08)
    gaps=[(float(b['preprocess_start'])-float(a['call_return']))*1000 for a,b in zip(charged,charged[1:])]
    observer={}
    for k in ('loop_guard_s','bookkeeping_s','status_write_s','log_write_s'):
        # Last batch's status/log dispatch is after the cost endpoint and excluded.
        selected=charged if k=='loop_guard_s' else charged[:-1]
        if any('observer_diagnostics' in c for c in selected):
            if k=='loop_guard_s':selected=selected[1:]  # first pre-call admission is outside the charged union
            observer[k]=sum(c['observer_diagnostics'][k] for c in selected)
    return dict(model=report['model'],resource_cap=report['cap'],repeat=report['repeat'],batch=report.get('batch',1),
        srcc=actual['srcc'],plcc=actual['plcc'],time16_ms=seconds*80,energy16_j=result['energy_j']*.08,
        model_call_s=model_s,model_call16_ms=model_s*80,nonpreprocessing_gap16_ms=gap_s*80,
        charged_s=seconds,gpu_energy_j=gpu,video_evaluations=200,visible_sms=report['visible_multiprocessors'],
        actual_busy_percent=statistics.mean(float(r['utilization_percent']) for r in raw),
        checkpoint_sha256=report['checkpoint_sha256'],parameter_dtypes=report['floating_parameter_dtypes'],
        readout_dtypes=report['quality_readout_dtypes'],source=(folder/'report.json').relative_to(REPO).as_posix(),
        source_sha256=sha(folder/'report.json'),identity_sha256=sha(folder/'identity.json'),source_commit=report['git_commit'],
        completed_at_beijing=datetime.fromtimestamp(status.get('unix',status.get('time')),timezone(timedelta(hours=8))).isoformat(),
        calls_sha256=sha(folder/('calls.json' if json_calls else 'calls.csv')),telemetry_sha256=sha(folder/'telemetry.csv'),
        original_time16_ms=report['cost']['time16_ms'],original_energy16_j=report['cost']['energy16_j'],
        original_block16_ms=sum(float(c['legacy_host_ms']) for c in charged)*.08,
        model_front16_ms=front_s*80,last_cpu_return16_ms=tail_s*80,
        cost_calls=len(charged),internal_gaps=len(gaps),last_test_batch=int(calls[-1].get('videos',1)),
        internal_gap_ms_max=max(gaps),internal_gap_ms_top5=sorted(gaps,reverse=True)[:5],
        observer_diagnostics_s=observer,observer_not_subtracted=True,accounting=PROTOCOL,
        max_power_sample_age_s=result['max_power_sample_age_s'])


def build(include_batch8):
    source=BASE/'batch6_01/batch6_paper.json'
    prior=read(source);data=deepcopy(prior)
    derived=BASE/'batch8_01'
    if include_batch8:
        completed=read(derived/'final_saved_audit.json')
        assert completed['suite_status']=='complete' and completed['mps_released'] and completed['audited_points']==12
    revised=[]
    for point in data['points']:
        if (point['model'],point['resource_cap'])==('qwen',100):
            continue  # entire approved anchor, its three audit repeats and provenance remain unchanged
        if include_batch8 and point['model']=='clip':
            paths=[derived/f"clip_b08_cap{point['resource_cap']}_cost_r{r:02d}/report.json" for r in (1,2,3)]
        else:
            paths=[REPO/r['source'] for r in point['repeats']]
            assert all(sha(p)==r['source_sha256'] for p,r in zip(paths,point['repeats']))
        group=[replay(p.parent) for p in paths]
        assert len(group)==3 and [r['repeat'] for r in group]==[1,2,3]
        assert all(r['checkpoint_sha256']==point['repeats'][0]['checkpoint_sha256'] for r in group)
        assert all(r['parameter_dtypes']==point['repeats'][0]['parameter_dtypes'] and
            r['readout_dtypes']==point['repeats'][0]['readout_dtypes'] for r in group)
        point.update(batch=group[0]['batch'],time16_ms=statistics.mean(r['time16_ms'] for r in group),
            energy16_j=statistics.mean(r['energy16_j'] for r in group),repeats=group,
            srcc_repeats=[r['srcc'] for r in group],measured_srcc=statistics.mean(r['srcc'] for r in group),
            cost_boundary=PROTOCOL,measurement_reuse='Original cold distinct-video records, independently re-integrated with model front and CPU return',
            cost_instrumentation_note='All observed status/logging/dispatch intervals retained; none subtracted.')
        if point['model']=='clip' and include_batch8:
            resident=read(derived/f"clip_resident_cap{point['resource_cap']}/batch08/report.json")
            point['resident_utilization_reference_percent']=resident['mean_utilization_percent']
        revised.extend(group)
    for point in data['points']:
        if point['model']=='clip' and include_batch8:
            ref=next(p for p in data['points'] if p['model']=='clip' and p['resource_cap']==100)
            point.update(srcc=statistics.mean(ref['srcc_repeats']),accuracy_reference_repeats=ref['srcc_repeats'],
                accuracy_reference_reports=[dict(source=r['source'],sha256=r['source_sha256']) for r in ref['repeats']])
    anchor=next(p for p in data['points'] if p['model']=='qwen' and p['resource_cap']==100)
    assert anchor==next(p for p in prior['points'] if p['model']=='qwen' and p['resource_cap']==100)
    data.update(repeats=[r for p in data['points'] for r in p['repeats']],accounting=PROTOCOL,
        prior_paper_source=dict(path=source.relative_to(REPO).as_posix(),sha256=sha(source)),
        accounting_exception='Qwen100 finalized historical compute-only cost retained, not a new matched-boundary point',
        all_logs_retained_no_estimated_subtraction=True,
        method='Three cold fresh processes; full558 distinct test videos; first200 costs normalized16/200; no explicit inference warmup; prepared-input model calls plus internal gaps')
    data['reaccount_source_sha256']=sha(Path(__file__))
    data['selection_batches']={p['model']:p['batch'] for p in data['points']}
    if include_batch8:
        data['batch_selection_audit']=dict(path=(derived/'final_saved_audit.json').relative_to(REPO).as_posix(),sha256=sha(derived/'final_saved_audit.json'))
    comparison=[]
    for model,batch in [('clip',6),('clip',10),('yolo',6),('yolo',10)]+([('clip',8)] if include_batch8 else []):
        for cap in (100,50):
            group=[replay(BASE/f'batch{batch}_01/{model}_b{batch:02d}_cap{cap}_cost_r{r:02d}') for r in (1,2,3)]
            comparison.append(dict(model=model,batch=batch,resource_cap=cap,repeats=group,
                time16_ms=statistics.mean(r['time16_ms'] for r in group),energy16_j=statistics.mean(r['energy16_j'] for r in group),
                time_repeats_ms=[r['time16_ms'] for r in group],energy_repeats_j=[r['energy16_j'] for r in group]))
    output=derived/('ready_input_paper.json' if include_batch8 else 'ready_input_preliminary.json')
    output.write_text(json.dumps(data,indent=2)+'\n')
    (derived/('ready_input_comparison.json' if include_batch8 else 'ready_input_comparison_preliminary.json')).write_text(json.dumps(comparison,indent=2)+'\n')
    return output,data,revised


def replace_clip_only(prior,groups,resident):
    """User-selected existing B10 display, with every non-CLIP point intact."""
    data=deepcopy(prior)
    for point in data['points']:
        if point['model']!='clip':continue
        cap=point['resource_cap'];group=groups[cap]
        assert len(group)==3 and [r['repeat'] for r in group]==[1,2,3]
        assert all(r['model']=='clip' and r['batch']==10 and r['resource_cap']==cap for r in group)
        assert all(r['checkpoint_sha256']==point['repeats'][0]['checkpoint_sha256'] and
            r['parameter_dtypes']==point['repeats'][0]['parameter_dtypes'] and
            r['readout_dtypes']==point['repeats'][0]['readout_dtypes'] for r in group)
        point.update(batch=10,repeats=group,time16_ms=statistics.mean(r['time16_ms'] for r in group),
            energy16_j=statistics.mean(r['energy16_j'] for r in group),
            srcc_repeats=[r['srcc'] for r in group],measured_srcc=statistics.mean(r['srcc'] for r in group),
            resident_utilization_reference_percent=resident[cap],cost_boundary=PROTOCOL,
            measurement_reuse='Existing21October CLIP B10 cold trials; prepared-input time and power replay reverified22October; no new GPU run')
    reference=next(p for p in data['points'] if p['model']=='clip' and p['resource_cap']==100)
    for point in data['points']:
        if point['model']=='clip':
            point.update(srcc=statistics.mean(reference['srcc_repeats']),accuracy_reference_repeats=reference['srcc_repeats'],
                accuracy_reference_reports=[dict(source=r['source'],sha256=r['source_sha256']) for r in reference['repeats']])
        else:
            assert point==next(p for p in prior['points'] if (p['model'],p['resource_cap'])==(point['model'],point['resource_cap']))
    data['repeats']=[r for p in data['points'] for r in p['repeats']]
    data['selection_batches']={p['model']:p['batch'] for p in data['points']}
    return data


def select_clip_batch10():
    source=BASE/'batch8_01/ready_input_paper.json'
    assert sha(source)==read(source.parent/'figure_receipt.json')['source_sha256']
    prior=read(source);assert prior['accounting']==PROTOCOL
    comparison_source=source.parent/'ready_input_comparison.json'
    comparison=read(comparison_source)
    target=BASE/'batch10_01';saved=read(target/'final_saved_audit.json')
    assert saved['status']=='passed' and saved['suite_status']=='complete' and saved['mps_released'] and saved['audited_points']==24
    groups={};resident={}
    for cap in (100,50):
        previous=next(p for p in comparison if (p['model'],p['batch'],p['resource_cap'])==('clip',10,cap))
        group=[replay(target/f'clip_b10_cap{cap}_cost_r{r:02d}') for r in (1,2,3)]
        for now,old in zip(group,previous['repeats']):
            for k in ('source_sha256','identity_sha256','calls_sha256','telemetry_sha256','checkpoint_sha256'):
                assert now[k]==old[k],k
            close(now['time16_ms'],old['time16_ms']);close(now['energy16_j'],old['energy16_j'])
            assert now['cost_calls']==20 and now['internal_gaps']==19 and now['last_test_batch']==8
        groups[cap]=group
        resident[cap]=saved['selections']['clip']['caps'][str(cap)]['resident_nvml_mean_percent']
    data=replace_clip_only(prior,groups,resident)
    data.update(prior_paper_source=dict(path=source.relative_to(REPO).as_posix(),sha256=sha(source)),
        clip_batch10_audit=dict(path=(target/'final_saved_audit.json').relative_to(REPO).as_posix(),sha256=sha(target/'final_saved_audit.json')),
        presentation_selection='User22October selected existing CLIP B10; all other points unchanged; no GPU remeasurement',
        reaccount_source_sha256=sha(Path(__file__)))
    output=target/'ready_input_selected_paper.json'
    with output.open('x') as f:json.dump(data,f,indent=2);f.write('\n')
    with (target/'ready_input_comparison.json').open('x') as f:json.dump(comparison,f,indent=2);f.write('\n')
    with (target/'clip10_ready_input_reverification.json').open('x') as f:
        json.dump(dict(status='passed',accounting=PROTOCOL,points=groups,
            previous_recalculation_matched=True,non_clip_points_unchanged=True,
            no_new_GPU_execution=True,source_sha256=sha(Path(__file__))),f,indent=2)
    return output,data,[r for g in groups.values() for r in g]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--include-batch8',action='store_true');p.add_argument('--select-clip-batch10',action='store_true');args=p.parse_args()
    if args.include_batch8 and args.select_clip_batch10:p.error('Select one presentation')
    output,data,revised=select_clip_batch10() if args.select_clip_batch10 else build(args.include_batch8)
    print(json.dumps(dict(status='passed',independently_reaccounted=len(revised),output=str(output),
        points=[{k:r[k] for k in ('model','resource_cap','batch','time16_ms','energy16_j')} for r in data['points']]),indent=2))

if __name__=='__main__':main()
