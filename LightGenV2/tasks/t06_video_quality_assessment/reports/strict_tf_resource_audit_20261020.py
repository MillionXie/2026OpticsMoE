"""Replay strict TF/wait intervals and each used GPU's raw power observations."""
from __future__ import annotations
from bisect import bisect_right
import csv
from datetime import datetime,timezone,timedelta
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
from . import precision_comparison_20261019 as common

REPO=common.REPO;TASK=common.TASK
ROOT=TASK/'runs/simulation/lgvq_strict_tf_rental_dual_20261020'
LAB_ROOT=TASK/'runs/simulation/lgvq_busy_resource_lab_20261020'/'per_forward_first_block_plus_internal_wait_v1'/'recovery_1102'
LAB_CONTINUATION=LAB_ROOT.parent/'calibration_1140'
LAB_DURATION=LAB_ROOT.parent/'duration_1155'
PROTOCOL='per_forward_first_block_plus_internal_wait_v1'
ACCOUNTING='per_forward_tf_plus_output_to_next_input_v2'
NAMES={'deepseek':'DeepSeek-VL2','kimi':'Kimi-VL-A3B','qwen':'Qwen3-VL-2B','glm':'GLM-4.6V-Flash','clip':'CLIP ViT-B/32','yolo':'YOLO11s'}


def read(p):return json.loads(p.read_text())
def rows(p):return list(csv.DictReader(p.open(newline='',encoding='utf-8-sig')))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def frame_positions(record):
    raw=record.get('frame_positions',record.get('selected_frame_positions'))
    assert raw is not None
    frames=json.loads(raw)
    return frames[0] if len(frames)==1 and isinstance(frames[0],list) else frames
def close(a,b):
    if not math.isclose(float(a),float(b),abs_tol=1e-7,rel_tol=1e-10):raise AssertionError((a,b))


def observed_energy(intervals,raw):
    """Independent replay from raw CSV, not reported energy/average power."""
    stamps=[float(v['host_monotonic_s']) for v in raw];total=0.
    assert all(a<b for a,b in zip(stamps,stamps[1:]))
    for start,end in intervals:
        index=bisect_right(stamps,start)-1
        assert index>=0
        current=start
        while current<end:
            next_stamp=min(end,stamps[index+1] if index+1<len(stamps) else end)
            assert next_stamp-stamps[index]<=.25
            total+=float(raw[index]['watts'])*(next_stamp-current)
            current=next_stamp;index+=1
    return total


def charged_intervals(calls):
    """Exclude every frontend; include all intercall gaps exactly once."""
    assert calls
    compute=[(float(c['first_block_start']),float(c['gpu_output'])) for c in calls]
    gaps=[(float(a['gpu_output']),float(b['call_start'])) for a,b in zip(calls,calls[1:])]
    assert all(a<b for a,b in compute) and all(a<=b for a,b in gaps)
    intervals=sorted(compute+[(a,b) for a,b in gaps if a<b])
    assert all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:]))
    return intervals,gaps


def check(point,condition):
    point=point.resolve()
    identity=read(point/'identity.json');folder=point/condition;report=read(folder/'report.json')
    assert report['status']=='complete' and report['timing_protocol']==PROTOCOL
    assert report['every_preblock_interval_excluded'] and report['internal_wait_included'] and report['terminal_wait_excluded']
    assert report['individually_synchronized_forwards'] and report['cost_intervals_are_not_contiguous_wall_window']
    assert len(report['repeats'])==3 and identity['timing_protocol']==PROTOCOL
    laboratory='gpu_uuids' not in identity
    if not laboratory:assert read(point/'boundary_smoke.json')['status']=='passed'
    for name,digest in identity['sources'].items():
        blob=subprocess.check_output(['git','show',identity['git_commit']+':'+name.replace('\\','/')],cwd=REPO)
        assert hashlib.sha256(blob).hexdigest()==digest
    if identity['precision']=='fp32':assert identity.get('parameter_dtypes',identity.get('floating_parameter_dtypes'))==['torch.float32']
    devices=[identity['gpu_uuid']] if laboratory else identity['gpu_uuids']
    sms=[identity['visible_multiprocessors']] if laboratory else identity['visible_multiprocessors']
    if identity['method']=='mps':
        expected=(108 if identity['cap']==100 else 54) if not laboratory else (108 if identity['cap']==100 else 54)
        assert sms==[expected]*len(devices)
    if not laboratory:
        assert identity['parameter_devices']==['cuda:0','cuda:1'] if len(devices)==2 else identity['parameter_devices']==['cuda:0']
    if len(devices)==2:assert identity['model']=='deepseek' and identity['precision']=='fp32'
    process_events=read(point/'process_audit.json');assert process_events
    own=str(identity['pid']);seen_own=False
    for event in process_events:
        allowed_pids={own}|{str(v) for v in event.get('servers',[])}
        assert {str(v) for v in event.get('clients',[])}<={own}
        for row in event['processes']:
            assert str(row[0]).strip() in allowed_pids
            assert str(row[1]).strip() in devices
            seen_own|=str(row[0]).strip()==own
    assert seen_own
    calls=rows(folder/'calls.csv');waits=rows(folder/'waits.csv')
    calibration=report.get('pacing_calibration','short_feedback60s_v1')
    if calibration=='full_window_validation_v1':
        assert identity['pacing_calibration']==calibration and condition=='busy50'
        validations=[u for u in read(folder/'controller_updates.json') if u.get('stage')==calibration]
        assert 1<=len(validations)<=3 and validations[-1]['passed']
        assert abs(validations[-1]['observed']-50)<=2
        close(validations[-1]['old_duty'],report['frozen_duty'])
        assert (folder/'calibration_calls.csv').exists() and (folder/'calibration_waits.csv').exists()
        for update in validations:
            raw_validation=read(folder/f"calibration_validation{update['attempt']:02d}.json")
            assert raw_validation['not_formal_measurement']
            assert raw_validation['wall_end']<report['repeats'][0]['wall_start']
            close(raw_validation['observed'],update['observed'])
            close(raw_validation['duty'],update['old_duty'])
    streams=[]
    for i,gpu in enumerate(devices):
        suffix='' if i==0 else f'_gpu{i:02d}'
        raw=rows(folder/f'telemetry{suffix}.csv');events=[json.loads(s) for s in (folder/f'telemetry{suffix}_stdout.jsonl').read_text().splitlines()]
        assert raw and len(raw)==len(events) and all(e['valid_numeric_row'] for e in events)
        assert all(math.isfinite(float(v['watts'])) and float(v['watts'])>0 and 0<=float(v['utilization_percent'])<=100 for v in raw)
        clocks=rows(folder/f'clock_telemetry{suffix}.csv');assert clocks and all(float(v['power_limit_w'])==(250 if laboratory else 400) for v in clocks)
        streams.append(raw)
    if calibration=='full_window_validation_v1':
        for update in validations:
            window=read(folder/f"calibration_validation{update['attempt']:02d}.json")
            observed=[]
            for raw in streams:
                values=[float(v['utilization_percent']) for v in raw
                        if window['wall_start']<=float(v['host_monotonic_s'])<window['wall_end']]
                assert len(values)>=100;observed.append(statistics.mean(values))
            close(max(observed),window['observed'])
    repeats=[]
    for r in report['repeats']:
        rc=[v for v in calls if int(v['repeat'])==r['repeat']];rw=[v for v in waits if int(v['repeat'])==r['repeat']]
        assert len(rc)==r['forward_calls']==r['groups']*8
        assert r['video_evaluations']==len(rc)*identity['batch']
        assert all(float(c['call_start'])<=float(c['first_block_start'])<float(c['gpu_output']) for c in rc)
        assert all(float(a['gpu_output'])<=float(b['call_start']) for a,b in zip(rc,rc[1:]))
        compute=sum(float(v['gpu_output'])-float(v['first_block_start']) for v in rc)
        for c in rc:
            close(float(c['first_block_start'])-float(c['call_start']),c['excluded_preblock_s'])
            close(float(c['gpu_output'])-float(c['first_block_start']),c['compute_s'])
        assert len(rw)==(r['groups']-1 if report['frozen_duty']<1 else 0)
        for w in rw:
            g=int(w['group']);assert g<r['groups']-1
            assert float(rc[g*8+7]['gpu_output'])<=float(w['start'])<float(w['end'])<=float(rc[(g+1)*8]['call_start'])
            assert float(w['end'])-float(w['start'])>=float(w['requested_s'])-.001
        waiting=sum(float(v['end'])-float(v['start']) for v in rw)
        intervals=sorted([(float(c['first_block_start']),float(c['gpu_output'])) for c in rc]+
                         [(float(w['start']),float(w['end'])) for w in rw])
        assert all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:]))
        close(compute,r['compute_s']);close(waiting,r['internal_wait_s']);close(compute+waiting,r['elapsed_s']);assert r['elapsed_s']>=30
        assert r['elapsed_s']>=report.get('minimum_charged_s',0.)-1e-8
        assert r['wall_start']<=intervals[0][0] and intervals[-1][1]<=r['wall_end']
        board=[observed_energy(intervals,raw) for raw in streams]
        energy=338.2*(compute+waiting)+sum(board);close(energy,r['energy_j'])
        infos=[]
        for gpu,raw in zip(devices,streams):
            selected=[v for v in raw if r['wall_start']<=float(v['host_monotonic_s'])<r['wall_end']]
            info=r['per_gpu_full_wall_telemetry'][gpu];assert len(selected)==info['samples']
            close(statistics.mean(float(v['utilization_percent']) for v in selected),info['mean_gpu_utilization_percent'])
            close(statistics.mean(float(v['watts']) for v in selected),info['mean_board_power_w'])
            infos.append(info)
        close((compute+waiting)*16000/r['video_evaluations'],r['time16_ms']);close(energy*16/r['video_evaluations'],r['energy16_j'])
        # Latest author request: also charge output->NEXT model-input entry.
        # Explicit pacing waits are already inside these gaps: do NOT add twice.
        new_intervals,gaps=charged_intervals(rc)
        gap=sum(b-a for a,b in gaps);assert gap>=waiting-1e-7
        next_tf_gap=sum(float(b['first_block_start'])-float(a['gpu_output']) for a,b in zip(rc,rc[1:]))
        cycle_span=float(rc[-1]['gpu_output'])-float(rc[0]['first_block_start'])
        close(cycle_span,compute+next_tf_gap)
        board=[observed_energy(new_intervals,raw) for raw in streams]
        new_elapsed=compute+gap;new_energy=338.2*new_elapsed+sum(board)
        repeats.append(dict(r,gpu_energy_j=sum(board),per_gpu_board_energy_j=board,
            recorded_wait_only_elapsed_s=r['elapsed_s'],recorded_wait_only_energy_j=r['energy_j'],
            recorded_time16_ms=r['time16_ms'],recorded_energy16_j=r['energy16_j'],
            elapsed_s=new_elapsed,energy_j=new_energy,time16_ms=new_elapsed*16000/r['video_evaluations'],
            energy16_j=new_energy*16/r['video_evaluations'],accounting_protocol=ACCOUNTING,
            output_to_next_model_input_s=gap,other_intercall_gap_s=max(0.,gap-waiting),
            output_to_next_tf_s=next_tf_gap,in_cycle_preblock_s=next_tf_gap-gap,full_cycle_span_s=cycle_span,
            first_tf_start=float(rc[0]['first_block_start']),final_gpu_output=float(rc[-1]['gpu_output']),gpu_infos=infos))
    per_gpu=[]
    for i,gpu in enumerate(devices):
        pooled=sum(r['gpu_infos'][i]['samples']*r['gpu_infos'][i]['mean_gpu_utilization_percent'] for r in repeats)/sum(r['gpu_infos'][i]['samples'] for r in repeats)
        close(pooled,report['per_gpu_busy_percent'][gpu]);per_gpu.append(pooled)
    hot=per_gpu.index(max(per_gpu));close(max(per_gpu),report['mean_gpu_busy_percent'])
    for r in repeats:r['measurement']=r['gpu_infos'][hot]
    videos=sum(r['video_evaluations'] for r in repeats);elapsed=sum(r['elapsed_s'] for r in repeats);energy=sum(r['energy_j'] for r in repeats)
    close(sum(r['recorded_wait_only_elapsed_s'] for r in repeats)*16000/videos,report['time16_ms'])
    close(sum(r['recorded_wait_only_energy_j'] for r in repeats)*16/videos,report['energy16_j'])
    acquired=next(v['host_time_unix'] for v in streams[0] if float(v['host_monotonic_s'])>=repeats[0]['wall_start'])
    return dict(model=identity['model'],model_name=NAMES[identity['model']],precision=identity['precision'],method=identity['method'],condition=condition,
        batch=identity['batch'],resource_cap=identity['cap'] if identity['method']=='mps' else None,visible_sms=' + '.join(map(str,sms)),
        gpu_count=len(devices),gpu_uuids=devices,hot_gpu_index=hot,per_gpu_busy_percent=per_gpu,busy_percent=max(per_gpu),
        allocation_average_busy_percent=statistics.mean(per_gpu),repeats=repeats,time16_ms=elapsed*16000/videos,energy16_j=energy*16/videos,
        frozen_duty=report['frozen_duty'],pacing_calibration=calibration,
        interval_mean_board_power_w=sum(r['gpu_energy_j'] for r in repeats)/elapsed,
        per_gpu_interval_mean_board_power_w=[sum(r['per_gpu_board_energy_j'][i] for r in repeats)/elapsed for i in range(len(devices))],
        accounting_protocol=ACCOUNTING,
        measured_at=datetime.fromtimestamp(float(acquired),timezone(timedelta(hours=8))).isoformat(),
        checkpoint_sha256=identity['checkpoint_sha256'],assets=identity['assets'],
        model_parameters=identity.get('original_parameters'),
        first_block_module={'deepseek':'vision.blocks.0','kimi':'vision_tower.encoder.blocks.0','qwen':'model.visual.blocks.0','glm':'model.visual.blocks.0','clip':'visual.transformer.resblocks.0','yolo':'model.0'}[identity['model']],
        platform='A100-PCIE-40GB /250W' if laboratory else f'{len(devices)} × A100-SXM4-80GB /400W per used GPU',source=(folder/'report.json').relative_to(REPO).as_posix(),
        source_sha256=sha(folder/'report.json'),status='measured' if report['target50_reached'] is not False else 'outside50±5pp',
        accuracy_note='No fresh accuracy in resident cost window; reference is separately identified.')


def verify_accuracy(path,head_sha):
    import numpy as np
    from scipy.stats import spearmanr,kendalltau,pearsonr
    report=read(path);raw=rows(path.with_name('predictions.csv'))
    assert len(raw)==558 and [int(r['sample_index']) for r in raw]==list(range(558))
    assert len({r['sample_id'] for r in raw})==558
    manifest=ROOT/'transfer/lgvq_train2250_test558.csv'
    assert sha(manifest)=='607c50d20662a47795c23cd2038081b30368ed108ba9be8a1bb8a9d250f8e7fc'
    original=[r for r in rows(manifest) if r['split']=='test']
    assert len(original)==558
    for record,expected in zip(raw,original):
        assert record['sample_id']==expected['sample_id']
        close(record['target_mos'],expected['temporal'])
        frames=frame_positions(record)
        assert len(frames)==4 and frames==sorted(frames)
    y=np.array([float(r['target_mos']) for r in raw]);p=np.array([float(r['prediction']) for r in raw])
    assert np.isfinite(y).all() and np.isfinite(p).all()
    metrics=dict(srcc=float(spearmanr(y,p).statistic),plcc=float(pearsonr(y,p).statistic),krcc=float(kendalltau(y,p).statistic),
                 rmse=float(np.sqrt(np.mean((p-y)**2))),mae=float(np.mean(np.abs(p-y))))
    actual=report.get('metrics',report.get('performance'));assert actual is not None
    for k,v in metrics.items():close(v,actual[k])
    assert report.get('checkpoint_sha256',report.get('identity',{}).get('checkpoint_sha256'))==head_sha
    return dict(srcc_reference=metrics['srcc'],accuracy_source=path.relative_to(REPO).as_posix(),accuracy_source_sha256=sha(path),
        accuracy_predictions_sha256=sha(path.with_name('predictions.csv')),accuracy_manifest_sha256=sha(manifest),
        accuracy_note='Independent frozen full558 TEST development estimate at the same batch/precision; not a fresh restricted-cap evaluation.')


def main():
    points=[]
    for root in (ROOT,ROOT/'attempt02',ROOT/'native_tail01',ROOT/'native_tail02',LAB_ROOT,LAB_CONTINUATION,LAB_DURATION):
        if not root.exists():continue
        for folder in root.iterdir():
            if not folder.is_dir() or not (folder/'identity.json').exists() or not (folder/'status.json').exists():continue
            if read(folder/'status.json')['status']!='complete':continue
            for condition in ('mps_cap100','mps_cap50','unrestricted','busy50'):
                if (folder/condition/'report.json').exists():points.append(check(folder,condition))
    keys=[(p['model'],p['precision'],p['method'],p['condition']) for p in points];assert len(keys)==len(set(keys))
    for model in NAMES:
        same=[p for p in points if p['model']==model]
        assert len({p['checkpoint_sha256'] for p in same})<=1
        assert len({json.dumps(p['assets'],sort_keys=True) for p in same})<=1
    for model in ('deepseek','kimi'):
        for precision in ('native','fp32'):
            for method in ('temporal','mps'):
                same=[p for p in points if (p['model'],p['precision'],p['method'])==(model,precision,method)]
                assert len({p['batch'] for p in same})<=1
                assert len({tuple(p['gpu_uuids']) for p in same})<=1
    old=read(common.OLD);optical=old['groups']['busy50_near_saturation'][0]
    refs={}
    for group in ('batch1_near_saturation','busy50_near_saturation'):
        for r in old['groups'][group]:
            if r.get('batch') and r.get('srcc') is not None:
                refs[('deepseek' if r['baseline']=='full' else r['baseline'],r['batch'])]=r.get('accuracy_source') or r.get('cost_source')
    for p in points:
        if p['precision']=='native':path=[REPO/s.strip() for s in refs[(p['model'],p['batch'])].split(';')]
        elif p['model']=='kimi':path=TASK/'runs/simulation/lgvq_busy_resource_20261020/accuracy_kimi_fp32_batch1/evaluation/report.json'
        elif p['model']=='deepseek':path=next((root/'mps_deepseek_fp32_100/accuracy558/report.json' for root in (ROOT/'attempt02',ROOT) if (root/'mps_deepseek_fp32_100/accuracy558/report.json').exists()),None)
        else:
            path=next((root/f'accuracy_fp32/{p["model"]}_batch{p["batch"]:02d}/report.json'
                       for root in (LAB_DURATION,LAB_CONTINUATION,LAB_ROOT)
                       if (root/f'accuracy_fp32/{p["model"]}_batch{p["batch"]:02d}/report.json').exists()),None)
        paths=path if isinstance(path,list) else ([path] if path is not None else [])
        if paths:
            if p['precision']=='native':assert all(v.exists() for v in paths),paths
            if not all(v.exists() for v in paths):continue
            verified=[verify_accuracy(v,p['checkpoint_sha256']) for v in paths]
            p.update(verified[0]);p['srcc_reference']=statistics.mean(v['srcc_reference'] for v in verified)
            p['accuracy_reference_count']=len(verified)
            for field in ('accuracy_source','accuracy_source_sha256','accuracy_predictions_sha256'):
                p[field]='; '.join(v[field] for v in verified)
            if len(verified)>1:p['accuracy_note']='Mean of ALL matching-batch full558 frozen development evaluations; all references and predictions independently audited. Not restricted-cap accuracy.'
            identity=read((REPO/p['source']).parent.parent/'identity.json')
            deltas=[]
            for reference_path in paths:
                reference=rows(reference_path.with_name('predictions.csv'))[:p['batch']]
                assert identity['sample_ids']==[r['sample_id'] for r in reference]
                assert identity['frame_positions']==[frame_positions(r) for r in reference]
                deltas.extend(abs(float(v)-float(r['prediction'])) for repeat in p['repeats'] for v,r in zip(repeat['output'],reference))
            p['resident_reference_max_abs_mos_delta']=max(deltas)
    target=ROOT/'presentation';target.mkdir(parents=True,exist_ok=True)
    rental_count=sum(p['model'] in ('deepseek','kimi') for p in points)
    active_lab=LAB_DURATION if LAB_DURATION.exists() else LAB_CONTINUATION if LAB_CONTINUATION.exists() else LAB_ROOT
    lab_state=read(active_lab/'suite_status.json') if (active_lab/'suite_status.json').exists() else {'status':'partial_local_backup'}
    data=dict(timing_protocol=PROTOCOL,accounting_protocol=ACCOUNTING,status='all48_complete' if len(points)==48 else 'rental16_complete' if rental_count==16 else 'partial',values=points,optical=optical,models=NAMES,
        rental_count=rental_count,laboratory_count=len(points)-rental_count,
        laboratory_status=lab_state['status'],capacity_missing='DeepSeek FP32 uses TWO80GB GPUs; all other cases use one. No CPU/disk offload.',
        contract='Sum EACH first-block-to-GPU-output interval plus output-to-NEXT-model-input gaps. Pacing waits are INCLUDED in gaps, never added twice. Exclude EACH frontend. Sum used-GPU board interval energies; host338.2W once. Full-wall NVML busy is separately reported. Also retain output-to-next-TF and full-cycle-span audit.',
        B1_historical_anchor_unchanged=True,accuracy_reference_not_per_control=True)
    (target/'eight_tables.json').write_text(json.dumps(data,indent=2)+'\n')
    columns=['model','model_parameters','precision','method','condition','batch','gpu_count','visible_sms','frozen_duty','pacing_calibration','busy_percent','per_gpu_busy_percent','allocation_average_busy_percent','time16_ms','energy16_j','interval_mean_board_power_w','per_gpu_interval_mean_board_power_w','accounting_protocol','first_block_module','srcc_reference','resident_reference_max_abs_mos_delta','platform','measured_at','source','source_sha256','accuracy_source','accuracy_source_sha256','accuracy_predictions_sha256','accuracy_manifest_sha256','accuracy_note']
    with (target/'paired_measurements.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(points)
    fields=['model','precision','method','condition','repeat','batch','gpu_count','elapsed_s','compute_s','output_to_next_model_input_s','internal_wait_s','other_intercall_gap_s','output_to_next_tf_s','in_cycle_preblock_s','excluded_preblock_s','full_cycle_span_s','wall_elapsed_s','video_evaluations','gpu_energy_j','energy_j','time16_ms','energy16_j','recorded_time16_ms','recorded_energy16_j','source','source_sha256']
    with (target/'paired_repeats.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for p in points:
            for r in p['repeats']:writer.writerow({**p,**r})
    print(json.dumps(dict(audited_points=len(points),windows=3*len(points),status=data['status'])))


if __name__=='__main__':main()
