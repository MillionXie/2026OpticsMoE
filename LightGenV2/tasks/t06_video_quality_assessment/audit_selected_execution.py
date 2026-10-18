"""Offline verification of the sparse run; no models, GPU, SSH or workbook edits."""
from __future__ import annotations
import ast
import csv
import hashlib
import json
import statistics
import subprocess
from collections import Counter
from datetime import datetime,timedelta,timezone
from math import sqrt,isfinite
from pathlib import Path

TASK=Path(__file__).resolve().parent
REPO=TASK.parents[2]
ROOT=TASK/'runs/simulation/lgvq_selected_execution_a100_20261016'
REPORT=REPO/'LightGenV2/reports/20260927_lgvq_temporal_multi_baseline_a100/recalculated_20261013'
SHA='607c50d20662a47795c23cd2038081b30368ed108ba9be8a1bb8a9d250f8e7fc'
NAMES=dict(qwen='Qwen3-VL-2B',glm='GLM-4.6V-Flash',small='DeepSeek-VL2-Small',clip='CLIP ViT-B/32',yolo='YOLO11s',deepseek='DeepSeek-VL2-Tiny')


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def csvrows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig',newline='')))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def ranks(values):
    order=sorted(range(len(values)),key=values.__getitem__);result=[0.]*len(values);start=0
    while start<len(order):
        end=start+1
        while end<len(order) and values[order[end]]==values[order[start]]:end+=1
        for i in order[start:end]:result[i]=(start+end+1)/2
        start=end
    return result


def pearson(a,b):
    ma,mb=statistics.mean(a),statistics.mean(b);a=[v-ma for v in a];b=[v-mb for v in b]
    return sum(x*y for x,y in zip(a,b))/sqrt(sum(x*x for x in a)*sum(y*y for y in b))


def guard(p,pid):
    rows=read(p);assert rows and all(all(int(v['pid'])==pid for v in r['processes']) for r in rows),str(p)


def same_outputs(eq):
    assert all(x['equal'] and x['max_abs']==0 and x['native_sha256']==x['candidate_sha256'] for x in eq['outputs'].values())


def work(ops):
    counter=Counter()
    for item in ops:
        item=dict(item,tensors=[dict(v) for v in item['tensors']]);count=1
        if 'scaled_dot_product' in item['op']:
            count=item['tensors'][0]['shape'][0]
            for tensor in item['tensors'][:3]:tensor['shape']=tensor['shape'][1:]
        counter[json.dumps(item,sort_keys=True)]+=count
    return counter


def native_history_comparison(output):
    """Audit old/new raw complete windows, not a causal thermal experiment."""
    runs=TASK/'runs/simulation'
    old=runs/'lgvq_utilization_execution_20261015/unified_repeat_suite01/remaining_native01'
    high=runs/'lgvq_residual_sync_a100_20261016/optimization01/clip_recovery02/clip'
    latest=runs/'lgvq_selected_execution_a100_20261016'
    entries=[('YOLO old sequential B3','yolo',3,old/'yolo/batch03',old/'yolo/identity.json','repeat'),
        ('YOLO selected fresh B3','yolo',3,latest/'yolo_batch03/util/batch03',latest/'yolo_batch03/util/identity.json','repeat'),
        ('CLIP old sequential B10','clip',10,old/'clip/batch10',old/'clip/identity.json','repeat'),
        ('CLIP native control after earlier work B10','clip',10,high/'batch10',high/'identity.json','control_repeat'),
        ('CLIP selected fresh B10','clip',10,latest/'clip_batch10/util/batch10',latest/'clip_batch10/util/identity.json','repeat')]
    for row in output['paired_rows']:
        folder=ROOT/f"{row['baseline']}_batch{row['batch_size']:02d}/util"
        entries.append((f"{row['model']} prescribed recheck B{row['batch_size']}",row['baseline'],row['batch_size'],folder/f"batch{row['batch_size']:02d}",folder/'identity.json','repeat'))
    methods=['unified_resident_utilization.py','evaluate_temporal_baseline_batches.py','audit_utilization_execution.py','batch_telemetry.py']
    identical={};window_rows=[];files={};cohorts=[]
    for label,model,batch,folder,identity_path,prefix in entries:
        identity=read(identity_path);files[identity_path.relative_to(REPO).as_posix()]=sha(identity_path)
        assert identity['checkpoint_sha256']==next(x['checkpoint_sha256'] for x in output['paired_rows'] if x['baseline']==model)
        for name in methods:
            key=(TASK/name).relative_to(REPO).as_posix();h=identity['source_sha256'][key]
            if name in identical:assert identical[name]==h
            identical[name]=h
        vals=[]
        for repeat in [1,2,3]:
            p=folder/f'{prefix}{repeat:02d}';report=read(p/'report.json');w=report['measurement']
            raw=csvrows(p/'telemetry.csv');raw=[x for x in raw if w['start_monotonic_s']<=float(x['host_monotonic_s'])<w['end_monotonic_s']]
            u=[float(x['utilization_percent']) for x in raw];assert len(u)==w['samples']
            assert w['duration_s']>=30 and report['warmup']['duration_s']>=5
            assert abs(statistics.mean(u)-w['mean_gpu_utilization_percent'])<1e-10
            clocks=csvrows(p/'clock_telemetry.csv');clocks=[x for x in clocks if w['start_monotonic_s']<=float(x['host_monotonic_s'])<w['end_monotonic_s']]
            row=dict(cohort=label,baseline=model,batch=batch,repeat=repeat,mean_percent=statistics.mean(u),samples=len(u),duration_s=w['duration_s'],forwards_per_second=report['measured_forward_invocations']/w['duration_s'],sm_clock_mhz=statistics.mean(float(x['sm_clock_mhz']) for x in clocks),temperature_c=statistics.mean(float(x['temperature_c']) for x in clocks),cpu_queue_wait_ms=(report['cpu_measurement_end']['main_thread_schedstat'][1]-report['cpu_measurement_start']['main_thread_schedstat'][1])/1e6,source=p.relative_to(REPO).as_posix())
            window_rows.append(row);vals.extend(u)
            for name in ['report.json','telemetry.csv','clock_telemetry.csv']:files[(p/name).relative_to(REPO).as_posix()]=sha(p/name)
        cohorts.append(dict(label=label,mean_percent=statistics.mean(vals),repeat_means=[x['mean_percent'] for x in window_rows[-3:]]))
    a=read(old/'yolo/batch03/report.json');b=read(latest/'yolo_batch03/util/batch03/report.json')
    equal={k:a[k]==b[k] for k in ['sample_ids','frame_positions','input_tensor_identity','reference_mos']};assert all(equal.values())
    current_input_checks={}
    for model,batch in [('yolo',3),('clip',10)]:
        a=read(latest/f'{model}_batch{batch:02d}/util/batch{batch:02d}/report.json')
        b=read(ROOT/f'{model}_batch{batch:02d}/util/batch{batch:02d}/report.json')
        current_input_checks[model]={k:a[k]==b[k] for k in ['sample_ids','frame_positions','input_tensor_identity','reference_mos']}
        assert all(current_input_checks[model].values())
    evidence=dict(core_module_sha256_identical=identical,yolo_original_selected_input_identity_equal=equal,recheck_same_batch_input_checks=current_input_checks,cohorts=cohorts,windows=window_rows,source_files_sha256=files,thermal_causality_is_not_isolated=True)
    (ROOT/'method_comparison.json').open('x',encoding='utf-8').write(json.dumps(evidence,indent=2)+'\n')
    with (ROOT/'method_comparison.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(window_rows[0]));writer.writeheader();writer.writerows(window_rows)
    lines=['### YOLO/CLIP reproducibility check (17 October 2026)','',
        'The inference-loop/statistics modules have identical SHA256 across the compared native runs: resident inputs prepared once, same-stream queue8, three complete ≥5-s warmups and ≥30-s measurement windows, all 50-ms samples pooled. YOLO B3 sample IDs, frames, resident tensor identity and reference outputs also match.',
        '',
        'The full execution histories were not identical. The old YOLO B3 followed B1/B2 in one loaded process. The old high CLIP B10 native control followed earlier batches/controls in one process. The recent selected run and this prescribed recheck start a fresh process for each point/phase. Clock, temperature and launch/host execution state therefore need not match despite an identical inner loop. The prior statement “completely identical” applied only to the loop/statistics, not this process history.',
        '',
        '| Native record | Repeat1 (%) | Repeat2 (%) | Repeat3 (%) | Pooled mean (%) |',
        '| --- | ---: | ---: | ---: | ---: |']
    for x in cohorts:lines.append('| '+x['label']+' | '+' | '.join(f'{v:.5f}' for v in x['repeat_means'])+f" | {x['mean_percent']:.5f} |")
    lines+=['','The recheck uses the unchanged fresh-process method, not selected high windows or longer conditioning. If YOLO3/CLIP10 pooled means remain below98%, only the predeclared adjacent YOLO4/CLIP11 is added. Latest complete prescribed results replace the corresponding old displayed point regardless of direction. Earlier data remain historical and are never pooled across rounds.',
        '', 'Clock/temperature co-variation is diagnostic evidence, not an isolated causal attribution. The recheck YOLO B3 repeat2 was near97% while repeats1/3 were near82%; main-thread run-queue waiting was only about14ms per30s, so neither pure thermal warmup nor CPU scheduling delay alone explains the spread. Host launch service, CPU/GPU frequency and driver-window effects remain possible contributors. We do not claim to have eliminated all environmental variance. The GPU utilization counter is kernel-busy time, not compute occupancy and not a batch-linear quantity.',
        '', 'The per-repeat clocks, temperatures, forward throughput and CPU scheduling evidence are retained in T06 canonical recheck run `method_comparison.csv` with source SHA receipts. Precision, weights, original four frames, clock/power settings and finalized Qwen B1 are unchanged. Utilization is independent of the newly paired first200 latency/power cost measurements.','']
    (ROOT/'method_comparison.md').open('x',encoding='utf-8').write('\n'.join(lines))


def audit_rental_extension():
    """Normalize independently rechecked supplementary hardware points, without replacing PCIe data."""
    import importlib.util
    root=TASK/'runs/simulation/lgvq_full_deepseek_kimi_rental_20261017'
    delivery=root/'followup_delivery_20261017.json'
    assert sha(delivery)=='9d14e7ed61635f2a13ab766286d5bb04d82a960b8674ae9aea3b65d7e5802c95'
    # The received summary was exported by Windows with GBK text; hash raw bytes first.
    try:final=json.loads(delivery.read_text(encoding='utf-8-sig'))
    except UnicodeDecodeError:final=json.loads(delivery.read_text(encoding='gb18030'))
    assert final['status']=='complete_verified_and_gpu_released' and not final['unavailable_requested_points']
    for item in final['artifact_manifest']:assert sha(root/item['path'])==item['sha256']
    for key in ['cpu_audit','terminal_release','verified_backup_receipt']:
        item=final[key];assert sha(root/item['path'])==item['sha256']
    audit=read(root/final['cpu_audit']['path']);assert audit['status']=='passed' and not audit['partial'] and not audit['cuda_initialized']
    assert len(audit['results'])==8 and all(x['status']=='passed' for x in audit['results'].values())
    spec=importlib.util.spec_from_file_location('rental_cpu_auditor',TASK/'rental_followup_audit.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    manifest=root/'provisioning/original_manifest.csv';assert sha(manifest)==SHA
    test=[r for r in csvrows(manifest) if r['split']=='test'];assert len(test)==558
    original=read(root/'measurement_summary.json');old_models={x['key']:x for x in original['models']}
    # Every relevant original B1 artifact must match a retained transfer receipt.
    bound={}
    for receipt in root.glob('backup_receipt*.json'):
        for item in read(receipt).get('verified_files',[]):bound.setdefault(item['path'],set()).add(item['sha256'])
    definitions={'deepseek':('full','DeepSeek-VL2',27480134248,4.5),'kimi':('kimi','Kimi-VL-A3B-Instruct',16407656048,3.0),'qwen':('qwen80','Qwen3-VL-2B [80GB]',2_000_000_000,None)}
    normalized=[];repeats=[]
    for point in final['points']:
        model='qwen' if point['model'].startswith('Qwen/') else ('kimi' if point['model'].startswith('moonshotai/') else 'deepseek')
        baseline,name,total,active=definitions[model];batch=point['batch']
        if point.get('cost_source'):
            cost_folder=root/point['cost_source'];cost_folder=cost_folder.parent
            util_folder=(root/point['util_source']).parent
        else:
            cost_folder=root/model/'a100_batch01_cost';util_folder=root/model/'resident_batch01'
            assert sha(root/old_models[model]['audit_path'])==old_models[model]['audit_sha256']
        cost=read(cost_folder/'report.json');util=read(util_folder/'report.json')
        checked=helper.audit_cost(cost_folder,test,batch)
        for k,j in [('T16_ms','T16_ms'),('E16_j','E16_j'),('srcc','srcc'),('plcc','plcc')]:assert abs(checked[k]-point[j])<1e-9
        assert cost['identity']['checkpoint_sha256']==util['identity']['checkpoint_sha256']
        assert cost['identity']['gpu_uuid']==util['identity']['gpu_uuid']==point['gpu_uuid']
        assert cost['test_videos']==558 and cost['cost_videos']==200 and cost['explicit_warmup_forwards']==0 and cost['first_batch_included']
        assert cost['power']['rated_power_limit_w']==400 and cost['cost']['host_proxy_w']==338.2
        if batch==1 and model!='qwen':
            for file in list(cost_folder.glob('*.csv'))+[cost_folder/'report.json',util_folder/'report.json']+list(util_folder.glob('repeat*/*.json'))+list(util_folder.glob('repeat*/*.csv')):
                assert sha(file) in bound.get(file.relative_to(root).as_posix(),set()),file
        telemetry=csvrows(cost_folder/'telemetry.csv');tagged=[r for r in telemetry if r['phase'].startswith('active:')]
        timing=[r for r in csvrows(cost_folder/'batch_timing.csv') if r['in_cost_subset']=='True'];walls=[float(r['synchronized_wall_ms']) for r in timing]
        pred=csvrows(cost_folder/'predictions.csv');ys=[float(r['target_mos']) for r in pred];xs=[float(r['prediction']) for r in pred]
        means=[];samples=0;counter=0;duration=0;forwards=0;peaks=[];stamp=0
        for index in (1,2,3):
            folder=util_folder/f'repeat{index:02d}';d=read(folder/'report.json')
            selected=[r for r in csvrows(folder/'telemetry.csv') if d['start_monotonic_s']<=float(r['host_monotonic_s'])<d['end_monotonic_s']]
            values=[float(r['utilization_percent']) for r in selected];assert len(values)==d['samples'] and d['duration_s']>=30
            assert abs(statistics.mean(values)-d['mean_gpu_utilization_percent'])<1e-9
            groups=read(folder/'forward_groups.json');warm=[g for g in groups if g['phase']=='warmup'];assert warm[-1]['end']-warm[0]['start']>=5 and all(g['forwards']==8 for g in groups)
            calls=sum(g['forwards'] for g in groups if g['phase']=='measured');assert calls==d['measured_forwards']
            clocks=[r for r in csvrows(folder/'clock_telemetry.csv') if d['start_monotonic_s']<=float(r['host_monotonic_s'])<d['end_monotonic_s']]
            assert clocks and all(float(r['power_limit_w'])==400 for r in clocks)
            stamp=max(stamp,float(selected[-1]['host_time_unix']));means.append(statistics.mean(values));samples+=len(values);counter+=sum(values);duration+=d['duration_s'];forwards+=calls;peaks.append(max(values))
            repeats.append(dict(model=name,baseline=baseline,batch=batch,repeat=index,mean_percent=means[-1],peak_percent=max(values),duration_s=d['duration_s'],samples=len(values),forwards=calls,power_w=d['mean_board_power_w'],memory_mib=d['mean_memory_used_mib'],warmup_s=warm[-1]['end']-warm[0]['start'],sm_clock_mhz=statistics.mean(float(r['sm_clock_mhz']) for r in clocks),temperature_c=statistics.mean(float(r['temperature_c']) for r in clocks),cpu_queue_wait_ms=None,measured_local=datetime.fromtimestamp(float(selected[-1]['host_time_unix']),timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S'),source=(folder/'report.json').relative_to(REPO).as_posix(),gpu_uuid=point['gpu_uuid'],hardware='A100-SXM4-80GB / 400W',role='Supplementary hardware; three complete native queue8 repeats'))
        assert abs(counter/samples-point['resident_utilization_percent'])<1e-9 and abs(counter/samples-util['pooled_mean_utilization_percent'])<1e-9
        sorted_walls=sorted(walls);q=(len(walls)-1)*.95;lo=int(q);p95=sorted_walls[lo]+(sorted_walls[min(lo+1,len(walls)-1)]-sorted_walls[lo])*(q-lo)
        row=dict(model=name,baseline=baseline,batch_size=batch,srcc=point['srcc'],plcc=point['plcc'],first200_srcc=pearson(ranks(xs[:200]),ranks(ys[:200])),time_16_videos_ms=point['T16_ms'],energy_16_videos_j=point['E16_j'],board_mean_w=checked['active_board_power_w'],cost_wall_sum_ms=sum(walls),cost_calls=len(walls),wall_mean_ms=statistics.mean(walls),wall_p95_ms=p95,active_samples=len(tagged),gpu_util_mean_percent=statistics.mean(float(r['utilization_percent']) for r in tagged),gpu_util_peak_percent=max(float(r['utilization_percent']) for r in tagged),memory_mean_mib=statistics.mean(float(r['memory_used_mib']) for r in tagged),memory_peak_mib=max(float(r['memory_used_mib']) for r in tagged),independent_resident_util_mean_percent=counter/samples,independent_resident_util_peak_percent=max(peaks),util_repeat_means=means,util_samples=samples,util_duration_s=duration,util_forward_calls=forwards,util_sum_percent_points=counter,measured_local=datetime.fromtimestamp(stamp,timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S'),cost_measured_local=datetime.fromtimestamp(float(telemetry[-1]['host_time_unix']),timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S'),checkpoint_sha256=cost['identity']['checkpoint_sha256'],source=(cost_folder/'report.json').relative_to(REPO).as_posix(),utilization_source=(util_folder/'report.json').relative_to(REPO).as_posix(),execution='Native frozen; supplementary A100-SXM4-80GB /400W',parameter_metadata=dict(model_total_parameters_billion=total/1e9,moe_active_parameters_billion=active,parameter_basis='Loaded total (Qwen official nominal); MoE active is declared nominal, not measured FLOPS',parameter_source='https://huggingface.co/'+point['model']),hardware_platform='A100-SXM4-80GB',power_limit_w=400,gpu_uuid=point['gpu_uuid'],hardware_session='original rental B1' if batch==1 and model!='qwen' else 'followup rental B2 and Qwen control',same_card_batch_ablation=False)
        normalized.append(row)
    assert len(normalized)==5 and len(repeats)==15
    helper.audit_inputs(root/'qwen_device_control_20261017');difference=helper.audit_device_difference(root);assert difference['max_abs_mos_difference']==0
    output=dict(status='complete',paired_rows=normalized,repeats=repeats,followup_summary_sha256=sha(delivery),raw_files_SHA_verified=len(final['artifact_manifest']),cpu_audit_SHA=final['cpu_audit']['sha256'],qwen_device_difference=difference,no_original_point_replacement=True,source='followup_delivery_20261017.json')
    target=root/'verified_figure5_extension.json'
    if target.exists():assert read(target)==output
    else:target.open('x',encoding='utf-8').write(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(status='complete',new_points=5,complete_repeats=15,all558_and200_recomputed=True,original_points_unchanged=True)))
    return output


def audit_kimi_clean_extension():
    """Replay the clean-card Kimi handoff locally, with no GPU or remote access."""
    from . import rental_followup_audit as helper
    root=TASK/'runs/simulation/lgvq_full_deepseek_kimi_rental_20261017'
    delivery=read(root/'kimi_clean_delivery_20261018.json')
    assert delivery['status']=='complete_and_released' and delivery['two_end_audits_agree']
    assert delivery['final_audit_sha256']=='b102de3ef69e1c8a35b0acc9b5d99b076615e13578f2df91b520bc78f02610c9'
    assert sha(root/delivery['final_audit'])==delivery['final_audit_sha256']
    assert sha(root/delivery['backup_receipt'])==delivery['backup_receipt_sha256']
    backup=read(root/delivery['backup_receipt']);assert backup['status']=='complete' and len(backup['files'])==120
    for item in backup['files']:
        assert sha(root/item['path'])==item['sha256'],item['path']
        assert (root/item['path']).stat().st_size==item['bytes']
    manifest=root/'provisioning/original_manifest.csv';assert sha(manifest)==SHA
    test=[r for r in csvrows(manifest) if r['split']=='test'];assert len(test)==558
    saved=read(root/delivery['final_audit'])
    replay=helper.audit_sequential_session(root,'kimi_clean_morning01_20261018',test,False)
    assert replay['status']=='passed' and replay['results']==saved['results']
    assert replay['highest_native_batch']==6 and replay['native_above98']
    run=root/'sequential_busy50_20261018/kimi_clean_morning01_20261018/kimi'
    template=next(r for r in read(root/'verified_figure5_extension.json')['paired_rows'] if r['baseline']=='kimi' and r['batch_size']==1)
    paired=[];repeats=[];trend=[];service=[]
    def resident(folder,batch,paced):
        raw=read(folder/'report.json');assert raw['status']=='complete' and raw['paced']==paced
        rows=csvrows(folder/'telemetry.csv');clocks=csvrows(folder/'clock_telemetry.csv')
        groups=read(folder/'forward_groups.json');rr=[];counter=0;samples=0;duration=0;calls=0;stamp=0;peak=0
        for r in raw['repeats']:
            index=r['repeat'];start,end=r['start_monotonic_s'],r['end_monotonic_s']
            selected=[s for s in rows if start<=float(s['host_monotonic_s'])<end]
            values=[float(s['utilization_percent']) for s in selected];assert len(values)==r['samples'] and end-start>=30
            assert abs(statistics.mean(values)-r['mean_gpu_utilization_percent'])<1e-9
            warm=[g for g in groups if g['phase']==f'warmup{index}'];assert warm[-1]['end_monotonic_s']-warm[0]['start_monotonic_s']>=5
            count=sum(g['forwards'] for g in groups if g['phase']==f'measured{index}');assert count==r['forward_calls']
            cs=[c for c in clocks if start<=float(c['host_monotonic_s'])<end];assert cs and all(float(c['power_limit_w'])==400 for c in cs)
            date=datetime.fromtimestamp(float(selected[-1]['host_time_unix']),timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')
            rr.append(dict(model=template['model'],baseline='kimi',batch=batch,repeat=index,mean_percent=statistics.mean(values),peak_percent=max(values),duration_s=end-start,samples=len(selected),forwards=count,power_w=statistics.mean(float(s['watts']) for s in selected),memory_mib=statistics.mean(float(s['memory_used_mib']) for s in selected),sm_clock_mhz=statistics.mean(float(c['sm_clock_mhz']) for c in cs),temperature_c=statistics.mean(float(c['temperature_c']) for c in cs),cpu_queue_wait_ms=None,warmup_s=warm[-1]['end_monotonic_s']-warm[0]['start_monotonic_s'],measured_local=date,source=(folder/f'repeat{index:02d}.json').relative_to(REPO).as_posix(),gpu_uuid=delivery['gpu_uuid'],hardware='A100-SXM4-80GB /400W',paced_service=paced))
            counter+=sum(values);samples+=len(values);duration+=end-start;calls+=count;stamp=max(stamp,float(selected[-1]['host_time_unix']));peak=max(peak,max(values))
        assert len(rr)==3 and abs(counter/samples-raw['pooled_mean_utilization_percent'])<1e-9
        return raw,rr,dict(counter=counter,samples=samples,duration=duration,calls=calls,stamp=stamp,peak=peak)
    for batch in range(1,7):
        folder=run/f'batch{batch:02d}';raw,rr,aggregate=resident(folder,batch,False)
        repeats.extend(rr);trend.append(dict(batch=batch,mean_percent=aggregate['counter']/aggregate['samples'],repeat_means=[r['mean_percent'] for r in rr],full558_accuracy_available=batch in (1,6),source=(folder/'report.json').relative_to(REPO).as_posix()))
        if batch not in (1,6):continue
        cost_folder=run/f'evaluate_batch{batch:02d}';cost=read(cost_folder/'report.json')
        checked=helper.audit_cost(cost_folder,test,batch);assert cost['identity']['gpu_uuid']==delivery['gpu_uuid']
        assert not cost['not_proven_exclusive_device'] and cost['power']['rated_power_limit_w']==400
        assert cost['identity']['checkpoint_sha256']==saved['checkpoint_sha256']
        pred=csvrows(cost_folder/'predictions.csv');ys=[float(r['target_mos']) for r in pred];xs=[float(r['prediction']) for r in pred]
        timings=[r for r in csvrows(cost_folder/'batch_timing.csv') if r['in_cost_subset']=='True'];walls=[float(r['synchronized_wall_ms']) for r in timings]
        telemetry=csvrows(cost_folder/'telemetry.csv');active=[r for r in telemetry if r['phase'].startswith('active:')]
        ordered=sorted(walls);q=(len(ordered)-1)*.95;lo=int(q);p95=ordered[lo]+(ordered[min(lo+1,len(ordered)-1)]-ordered[lo])*(q-lo)
        row=dict(template,batch_size=batch,srcc=checked['srcc'],plcc=checked['plcc'],first200_srcc=pearson(ranks(xs[:200]),ranks(ys[:200])),time_16_videos_ms=checked['T16_ms'],energy_16_videos_j=checked['E16_j'],board_mean_w=checked['active_board_power_w'],cost_wall_sum_ms=sum(walls),cost_calls=len(walls),wall_mean_ms=statistics.mean(walls),wall_p95_ms=p95,active_samples=len(active),gpu_util_mean_percent=statistics.mean(float(r['utilization_percent']) for r in active),gpu_util_peak_percent=max(float(r['utilization_percent']) for r in active),memory_mean_mib=statistics.mean(float(r['memory_used_mib']) for r in active),memory_peak_mib=max(float(r['memory_used_mib']) for r in active),independent_resident_util_mean_percent=aggregate['counter']/aggregate['samples'],independent_resident_util_peak_percent=aggregate['peak'],util_repeat_means=[r['mean_percent'] for r in rr],util_samples=aggregate['samples'],util_duration_s=aggregate['duration'],util_forward_calls=aggregate['calls'],util_sum_percent_points=aggregate['counter'],measured_local=datetime.fromtimestamp(aggregate['stamp'],timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S'),cost_measured_local=datetime.fromtimestamp(float(telemetry[-1]['host_time_unix']),timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S'),checkpoint_sha256=saved['checkpoint_sha256'],source=(cost_folder/'report.json').relative_to(REPO).as_posix(),utilization_source=(folder/'report.json').relative_to(REPO).as_posix(),execution='Native frozen; clean same-card A100-SXM4-80GB /400W; utilization B1 to6 in one model process',utilization_profile='Independent native queue8; same model process B1 to6, three complete repeats; no highest-window selection',gpu_uuid=delivery['gpu_uuid'],hardware_session='Clean rental session20261018 ecac2c49',same_card_batch_ablation=True)
        paired.append(row)
        paced_folder=run/f'busy50_batch{batch:02d}';pr,prr,pa=resident(paced_folder,batch,True)
        duration=pa['duration'];power=sum(r['power_w']*r['duration_s'] for r in prr)/duration
        t16=duration*16000/(pa['calls']*batch);e16=(338.2+power)*t16/1000
        assert abs(t16-saved['results'][f'busy50_batch{batch:02d}']['service_T16_ms'])<1e-8
        assert abs(e16-saved['results'][f'busy50_batch{batch:02d}']['service_E16_j'])<1e-8
        service.append(dict(model=template['model'],baseline='kimi',batch=batch,mean_busy_percent=pa['counter']/pa['samples'],repeat_means_percent=[r['mean_percent'] for r in prr],duration_s=duration,completed_forward_calls=pa['calls'],completed_video_evaluations=pa['calls']*batch,unique_videos=batch,board_mean_w=power,host_proxy_w=338.2,service_T16_ms=t16,service_E16_j=e16,full558_srcc_separate_source=row['source'],target50_reached=pr['target50_reached'],source=(paced_folder/'report.json').relative_to(REPO).as_posix(),gpu_uuid=delivery['gpu_uuid'],platform='A100-SXM4-80GB /400W',not_dataset_once_cost=True,calibration='complete compute-plus-idle group feedback,30s then fixed duty;3x5s warmup and complete>=30s measurement'))
    assert all(x['mean_percent']<=98 for x in trend[:-1]) and trend[-1]['mean_percent']>98
    old=csvrows(root/'kimi/a100_batch01_cost/predictions.csv');new=csvrows(run/'evaluate_batch01/predictions.csv')
    assert [(r['sample_id'],r['prediction']) for r in old]==[(r['sample_id'],r['prediction']) for r in new]
    result=dict(status='complete',paired_rows=paired,repeats=repeats,utilization_only_trend=trend,paced_service=service,raw_files_SHA_verified=len(backup['files']),delivery_sha256=sha(root/'kimi_clean_delivery_20261018.json'),cpu_audit_sha256=sha(root/delivery['final_audit']),historical_B1_all558_exact=True,non_kimi_data_unchanged=True,no_GPU_or_remote_access=True)
    target=root/'verified_kimi_clean_extension.json'
    if target.exists():assert read(target)==result
    else:target.open('x',encoding='utf-8').write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='complete',native_utilization_batches=6,complete_native_repeats=18,cost_and_accuracy_points=2,paced_service_points=2,raw_SHA_files=120)))
    return result


def audit_busy50_point(folder):
    """Independent raw-window replay, including all paced idle time and failures."""
    report,identity=read(folder/'report.json'),read(folder/'identity.json')
    assert folder.name==f"{identity['baseline']}_batch{identity['batch']:02d}"
    heads=next(ast.literal_eval(n.value) for n in ast.parse((TASK/'measure_frozen_batch_costs.py').read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CHECKPOINTS' for t in n.targets))
    assert identity['checkpoint_sha256']==heads[identity['baseline']]
    receipt=read(folder/'backup_receipt_check01.json')
    for item in receipt['verified_files']:assert sha(folder/item['path'])==item['sha256']
    assert report['status']=='complete' and len(report['repeats'])==3
    assert identity['gpu_uuid']=='GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55' and identity['manifest_sha256']==SHA
    assert identity['protocol']['protocol'].endswith('v2_complete_group_feedback')
    raw=csvrows(folder/'measurement/telemetry.csv');groups=read(folder/'measurement/forward_groups.json')
    clocks=csvrows(folder/'measurement/clock_telemetry.csv');assert clocks and all(float(c['power_limit_w'])==250 for c in clocks)
    for r in raw:
        values=[float(r[k]) for k in ('watts','utilization_percent','memory_used_mib')]
        assert all(map(isfinite,values)) and values[0]>0 and 0<=values[1]<=100 and values[2]>=0
    raw_stdout=folder/'measurement/telemetry_stdout.jsonl'
    if raw_stdout.exists():
        records=[json.loads(line) for line in raw_stdout.read_text().splitlines()]
        assert len(records)==len(raw) and all(r['valid_numeric_row'] for r in records)
        assert read(folder/'measurement/measurement_evidence_status.json')['status']=='recorded'
        for event,r in zip(records,raw):assert list(map(float,event['raw_line'].split(',')))==[float(r[k]) for k in ('watts','utilization_percent','memory_used_mib')]
    def close(a,b):assert abs(a-b)<=max(1e-8,abs(b)*1e-11),(a,b)
    duration=0;calls=0;power_duration=0;util_sum=0;samples=0;means=[]
    for index,r in enumerate(report['repeats'],1):
        w=r['measurement'];start,end=w['start_monotonic_s'],w['end_monotonic_s']
        selected=[x for x in raw if start<=float(x['host_monotonic_s'])<end]
        assert len(selected)==w['samples'] and end-start>=30
        busy=statistics.mean(float(x['utilization_percent']) for x in selected)
        watts=statistics.mean(float(x['watts']) for x in selected)
        close(busy,w['mean_gpu_utilization_percent']);close(watts,w['mean_board_power_w'])
        warm=[g for g in groups if g['phase']==f'warmup{index}'];assert warm[-1]['end_monotonic_s']-warm[0]['start_monotonic_s']>=5
        measured=[g for g in groups if g['phase']==f'measured{index}'];assert measured and all(g['forwards']==8 and g['duty']==report['frozen_duty'] for g in measured)
        n=sum(g['forwards'] for g in measured);assert n==r['forward_calls']
        for g in measured:
            assert start<=g['start_monotonic_s']<g['compute_end_monotonic_s']<=g['end_monotonic_s']<=end
            close(g['requested_idle_s'],(g['compute_end_monotonic_s']-g['start_monotonic_s'])*(1/g['duty']-1))
        t16=(end-start)*16000/(n*identity['batch']);close(t16,r['cost']['service_time_16_videos_ms'])
        close((338.2+watts)*t16/1000,r['cost']['estimated_energy_16_videos_j'])
        assert max(abs(a-b) for a,b in zip(r['output'],identity['native_reference_mos']))<=.0002
        duration+=end-start;calls+=n;power_duration+=watts*(end-start);util_sum+=busy*len(selected);samples+=len(selected);means.append(busy)
    for update in report['calibration_updates']:
        selected=[r for r in raw if update['sample_start_monotonic_s']<=float(r['host_monotonic_s'])<update['sample_end_monotonic_s']]
        assert len(selected)==update['sample_count'];close(statistics.mean(float(r['utilization_percent']) for r in selected),update['observed_percent'])
    watts=power_duration/duration;t16=duration*16000/(calls*identity['batch']);e16=(338.2+watts)*t16/1000
    close(util_sum/samples,report['mean_gpu_busy_percent']);close(t16,report['pooled_service_cost']['service_time_16_videos_ms']);close(e16,report['pooled_service_cost']['estimated_energy_16_videos_j'])
    guard(folder/'measurement/process_audit.json',identity['pid'])
    baseline=identity['baseline']
    return dict(model=NAMES[baseline],baseline=baseline,batch=identity['batch'],mean_busy_percent=util_sum/samples,
        repeat_means_percent=means,duration_s=duration,completed_forward_calls=calls,completed_video_evaluations=calls*identity['batch'],unique_videos=identity['batch'],
        board_mean_w=watts,host_proxy_w=338.2,service_T16_ms=t16,service_E16_j=e16,target50_reached=abs(util_sum/samples-50)<=5,
        source=(folder/'report.json').relative_to(REPO).as_posix(),gpu_uuid=identity['gpu_uuid'],platform='A100-PCIE-40GB /250W',
        not_dataset_once_cost=True,calibration='v2 complete groups since preceding update,30s then frozen duty;3x5s warmup/complete>=30s',
        checkpoint_sha256=identity['checkpoint_sha256'],telemetry_profile='durable raw stdout,fail-fast reader' if raw_stdout.exists() else 'original numeric reader; pre-failure complete point',numeric_samples=samples,
        frozen_duty_fraction=report['frozen_duty'],measured_imposed_wait_s=sum(g['requested_idle_s'] for g in groups if g['phase'].startswith('measured')))


def audit_busy50_service_extension():
    base=TASK/'runs/simulation/lgvq_resident_busy50_a100_20261017'
    new=base/'attempt03_remaining_20261018';old=base/'attempt02_20261018'
    final=read(new/'live_followup_final.json');assert final['all_expected_complete'] and final['expected_points']==8
    receipt=read(new/'backup_receipt_terminal_complete01.json')
    status_file=next(x for x in receipt['verified_files'] if x['path'].startswith('server_suite_status_'))
    assert sha(new/status_file['path'])==status_file['sha256'];state=read(new/status_file['path'])
    assert state['status']=='complete' and len(state['completed'])==8
    preserved=state['continuation'];assert len(preserved['preserved_points'])==4 and len(preserved['files_sha256'])==45
    for name,h in preserved['files_sha256'].items():
        actual=old/(f'server_suite_status_{h[:12]}.json' if name=='suite_status.json' else name)
        assert sha(actual)==h
    names=preserved['preserved_points']+state['completed']
    expected=['qwen_batch01','qwen_batch03','glm_batch01','glm_batch02','small_batch01','small_batch03','clip_batch01','clip_batch10','yolo_batch01','yolo_batch03','deepseek_batch01','deepseek_batch08']
    assert names==expected
    rows=[audit_busy50_point((old if i<4 else new)/name) for i,name in enumerate(names)]
    presentation=read(TASK/'runs/simulation/lgvq_full_deepseek_kimi_rental_20261017/figure5_kimi_refresh_20261018/presentation_selected.json')
    assert presentation['status']=='complete' and len(presentation['rows'])==19
    references={(r['baseline'],r['batch_size_videos']):r for r in presentation['rows'] if r['baseline']!='ours'}
    for r in rows:
        ref=references[r['baseline'],r['batch']]
        r.update(separate_full558_srcc=ref['measured_full558_srcc'],full558_srcc_separate_source=ref['cost_source'],full558_accuracy_remeasured_under_pacing=False)
    kimi=audit_kimi_clean_extension()
    assert kimi['status']=='complete' and len(kimi['paced_service'])==2
    for r in kimi['paced_service']:
        row=dict(r,separate_full558_srcc=float(references['kimi',r['batch']]['measured_full558_srcc']),full558_accuracy_remeasured_under_pacing=False,
                 telemetry_profile='rental last-complete-group feedback,not identical to original40GB v2 controller')
        rows.append(row)
    result=dict(status='complete',original40GB_points=12,preserved_original_points=names[:4],new_completed_points=names[4:],
        rows=rows,all_rows_have_three_full_repeats=True,no_model_GPU_or_remote_access=True,Qwen_B1_paper_anchor_unchanged=True,
        interpretation='Supplementary paced resident service costs including waits; not dataset-once costs, not optical spatial-occupancy equivalence')
    target=new/'verified_busy50_service_extension.json'
    if target.exists():assert read(target)==result
    else:target.open('x',encoding='utf-8').write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='complete',original40GB_points=12,new_points=8,Kimi_paced_points=2,total_service_rows=len(rows))))
    return result


def audit_sequential():
    """Utilization-only same-process reproduction; no cost substitution."""
    root=TASK/'runs/simulation/lgvq_yolo_clip_sequential_a100_20261017'
    cfg=read(TASK/'configs/baselines/yolo_clip_sequential_20261017.json')
    state=read(root/'suite_status.json');assert state['status']=='complete' and state['completed']==['yolo','clip']
    assert cfg['cases']=={'yolo':[1,2,3],'clip':[1,2,3,4,6,8,9,10]}
    checkpoint=next(ast.literal_eval(n.value) for n in ast.parse((TASK/'measure_frozen_batch_costs.py').read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CHECKPOINTS' for t in n.targets))
    verified={}
    for receipt in root.glob('backup_receipt*.json'):
        for item in read(receipt)['verified_files']:
            p=receipt.parent/item['path'];assert sha(p)==item['sha256'];verified[p.resolve()]=item['sha256']
    assert len(verified)>=180
    result=[];repeats=[];sources={};identities={}
    for model,batches in cfg['cases'].items():
        folder=root/model;identity=read(folder/'identity.json');receipt=read(folder/'sequential_driver_receipt.json');r=read(folder/'report.json')
        assert r['status']=='complete' and [x['batch'] for x in r['points']]==batches
        assert identity['checkpoint_sha256']==receipt['checkpoint_sha256']==checkpoint[model] and identity['manifest_sha256']==SHA
        assert identity['protocol']['required_terminal_batch']==cfg['target_batches'][model]
        assert identity['protocol']['batches']=={model:batches}
        identities[model]=identity['pid']
        for path,h in receipt['source_sha256'].items():
            assert state['source_sha256'][path]==h
            key=(path,h)
            if key not in sources:
                content=subprocess.check_output(['git','show',identity['source_commit']+':'+path],cwd=REPO)
                assert hashlib.sha256(content).hexdigest()==h;sources[key]=True
        for batch in batches:
            point=folder/f'batch{batch:02d}';pr=read(point/'report.json');means=[];total=0;n=0;duration=0;calls=0;peak=0;stamp=0
            assert pr['status']=='complete' and pr['batch']==batch and len(pr['sample_ids'])==len(set(pr['sample_ids']))==batch
            if batch==cfg['target_batches'][model]:
                old=read(TASK/f'runs/simulation/lgvq_yolo_clip_recheck_a100_20261017/{model}_batch{batch:02d}/util/batch{batch:02d}/report.json')
                assert all(pr[k]==old[k] for k in ['sample_ids','frame_positions','input_tensor_identity','reference_mos'])
            for repeat in [1,2,3]:
                p=point/f'repeat{repeat:02d}';d=read(p/'report.json');w=d['measurement'];raw=csvrows(p/'telemetry.csv')
                chosen=[x for x in raw if w['start_monotonic_s']<=float(x['host_monotonic_s'])<w['end_monotonic_s']]
                u=[float(x['utilization_percent']) for x in chosen]
                assert len(u)==w['samples'] and w['duration_s']>=30 and d['warmup']['duration_s']>=5
                mean=statistics.mean(u);assert abs(mean-w['mean_gpu_utilization_percent'])<1e-9
                assert all(abs(a-b)<.0002 for a,b in zip(d['output'],pr['reference_mos']))
                guard(p/'process_audit.json',identity['pid'])
                groups=read(p/'forward_groups.json');assert all(g['forwards']==8 for g in groups)
                assert sum(g['forwards'] for g in groups if g['phase']=='measured')==d['measured_forward_invocations']
                clocks=[x for x in csvrows(p/'clock_telemetry.csv') if w['start_monotonic_s']<=float(x['host_monotonic_s'])<w['end_monotonic_s']]
                assert clocks and all(float(x['power_limit_w'])==250 for x in clocks)
                for name in ['report.json','telemetry.csv','clock_telemetry.csv','forward_groups.json','process_audit.json']:assert (p/name).resolve() in verified
                repeats.append(dict(model=NAMES[model],baseline=model,batch=batch,repeat=repeat,mean_percent=mean,peak_percent=max(u),power_w=w['mean_board_power_w'],memory_mib=w['mean_memory_used_mib'],warmup_s=d['warmup']['duration_s'],measured_local=datetime.fromtimestamp(float(chosen[-1]['host_time_unix']),timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S'),samples=len(u),duration_s=w['duration_s'],forwards=d['measured_forward_invocations'],forwards_per_second=d['measured_forward_invocations']/w['duration_s'],sm_clock_mhz=statistics.mean(float(x['sm_clock_mhz']) for x in clocks),temperature_c=statistics.mean(float(x['temperature_c']) for x in clocks),cpu_queue_wait_ms=(d['cpu_measurement_end']['main_thread_schedstat'][1]-d['cpu_measurement_start']['main_thread_schedstat'][1])/1e6,pid=identity['pid'],source=(p/'report.json').relative_to(REPO).as_posix()))
                total+=sum(u);n+=len(u);means.append(mean)
                duration+=w['duration_s'];calls+=d['measured_forward_invocations'];peak=max(peak,max(u));stamp=max(stamp,float(chosen[-1]['host_time_unix']))
            assert abs(total/n-pr['mean_utilization_percent'])<1e-9
            result.append(dict(model=NAMES[model],baseline=model,batch=batch,pid=identity['pid'],repeat_means=means,pooled_mean_percent=total/n,peak_percent=peak,samples=n,duration_s=duration,forwards=calls,sum_percent_points=total,measured_local=datetime.fromtimestamp(stamp,timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S'),all_three_at_least98=all(x>=98 for x in means),pooled_at_least98=total/n>=98,target_point=batch==cfg['target_batches'][model],spread_pp=max(means)-min(means),source=(point/'report.json').relative_to(REPO).as_posix()))
    assert len(result)==11 and len(repeats)==33 and identities['yolo']!=identities['clip']
    # The only uniform change is terminal-batch orchestration, not the measured loop.
    name='LightGenV2/tasks/t06_video_quality_assessment/unified_resident_utilization.py'
    body=lambda s:ast.dump(next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='measure_repeat'))
    old=subprocess.check_output(['git','show','91ebe416c5e0696d8420104009f8164667dbd185:'+name],cwd=REPO,text=True)
    new=subprocess.check_output(['git','show',identity['source_commit']+':'+name],cwd=REPO,text=True)
    assert body(old)==body(new)
    name='LightGenV2/tasks/t06_video_quality_assessment/audit_utilization_execution.py'
    old_compute=subprocess.check_output(['git','show','91ebe416c5e0696d8420104009f8164667dbd185:'+name],cwd=REPO,text=True)
    new_compute=subprocess.check_output(['git','show',identity['source_commit']+':'+name],cwd=REPO,text=True)
    compute=lambda s:ast.dump(next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='make_gpu_forward'))
    assert compute(old_compute)==compute(new_compute)
    name='LightGenV2/tasks/t06_video_quality_assessment/evaluate_temporal_baseline_batches.py'
    assert subprocess.check_output(['git','show','91ebe416c5e0696d8420104009f8164667dbd185:'+name],cwd=REPO)==subprocess.check_output(['git','show',identity['source_commit']+':'+name],cwd=REPO)
    out=dict(status='complete',points=result,repeats=repeats,raw_files_SHA_verified=len(verified),published_source_checks=len(sources),one_PID_per_model=identities,original_measurement_loop_AST_equal=True,original_native_compute_AST_equal=True,original_loader_source_equal=True,no_cost_accuracy_or_Qwen_B1_change=True)
    (root/'verified_sequential.json').open('x',encoding='utf-8').write(json.dumps(out,indent=2)+'\n')
    lines=['### Same-process native reproduction (17 October 2026)','',
        'YOLO was loaded once for B=1,2,3. CLIP was loaded once for B=1,2,3,4,6,8,9,10. Every batch retains all three complete repeats. The terminal targets B3/B10 were required regardless of whether an earlier batch reached98%. No additional B4/B11 target or retry was substituted.',
        '',
        '| Model | B | Repeat1 (%) | Repeat2 (%) | Repeat3 (%) | Pooled mean (%) | All three ≥98% |',
        '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for x in result:lines.append('| '+x['model']+' | '+str(x['batch'])+' | '+' | '.join(f'{v:.5f}' for v in x['repeat_means'])+f" | {x['pooled_mean_percent']:.8f} | {x['all_three_at_least98']} |")
    lines+=['','The measured loop is AST-identical to the original: resident first-B four-frame inputs prepared once, native same-stream queue8, each repeat ≥5s warmup plus a complete ≥30s window, all50ms driver samples pooled. All repeat process guards use one PID per model. Source/config/PT/manifest hashes and target input tensor/reference-output identities were verified. No Graph, weight/precision change, clocks/power setting change, training or highest-repeat selection was introduced.',
        '', 'The earlier98.76% CLIP B10 native control was interleaved with a candidate implementation after previous batches. This run uses only the prescribed all-native ascending plan and does not claim an exact reproduction of that interleaved history. The same-process profile is disclosed rather than silently pooled with the previous fresh-process round. One three-repeat run supports within-run reproducibility, not a guarantee for every subsequent run or machine state.',
        '', 'This is utilization-only. Original per-batch accuracy, timing and energy remain from their separately identified verified cost runs. New resident power is never used to recompute those costs. The finalized historical Qwen B1 is neither executed nor changed. The previous fresh-process observations (including YOLO B3 at87.15912898%) remain preserved, not deleted or relabeled as invalid.',
        '', 'All11 batch points and33 complete repeats are retained in the canonical sequential run and raw-repeat table with source roles. Only requested YOLO B3/CLIP B10 replace their displayed utilization annotations. Other B1 measurements are predecessor audit, not replacements of finalized references.','']
    (root/'method_comparison.md').open('x',encoding='utf-8').write('\n'.join(lines))
    print(json.dumps(dict(status='complete',points=len(result),repeats=len(repeats),targets=[x for x in result if x['target_point']])))
    return out


def audit(selected_points=None,recheck=False):
    suite=read(ROOT/'suite_status.json') if (ROOT/'suite_status.json').exists() else None
    if not selected_points:assert suite and suite['status']=='complete'
    cfg=read(TASK/('configs/baselines/yolo_clip_recheck_20261017.json' if recheck else 'configs/baselines/selected_execution_20261016.json'))
    batches_by_model=cfg['cases'] if recheck else cfg['batches']
    expected=[f'{m}_batch{b:02d}' for m,bs in batches_by_model.items() for b in bs]
    if recheck:
        assert suite['status']=='complete'
        for model,bs in batches_by_model.items():
            assert f'{model}_batch{bs[0]:02d}' in suite['completed']
            has_extra=f'{model}_batch{bs[1]:02d}' in suite['completed']
            first=read(ROOT/f'{model}_batch{bs[0]:02d}/util/report.json')['points'][0]['mean']
            assert has_extra==(first<98)
        expected=suite['completed']
    if selected_points:assert set(selected_points)<=set(expected)
    checkpoints=next(ast.literal_eval(n.value) for n in ast.parse((TASK/'measure_frozen_batch_costs.py').read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CHECKPOINTS' for t in n.targets))
    verified={}
    for receipt in ROOT.rglob('backup_receipt*.json'):
        for row in read(receipt).get('verified_files',[]):
            p=receipt.parent/row['path'];assert sha(p)==row['sha256'];verified[p.resolve()]=row['sha256']
    assert len(verified)>=(20 if selected_points else 100)
    old=read(TASK/'runs/simulation/lgvq_frozen_batch_costs_a100_20261015/verified_costs.json')
    old_plot=csvrows(REPORT/'plot_data.csv');catalog={k:next(x for x in old_plot if x['baseline']==k) for k in NAMES}
    anchor=next(x for x in old_plot if x['baseline']=='qwen' and x['batch_size_videos']=='1')
    assert [float(anchor[k]) for k in ['time_16_videos_ms','energy_16_videos_j','display_gpu_util_mean_percent','measured_full558_srcc']]==[1188.3861464262009,462.1134010767563,63.133111480865225,.7663428036522781]
    result=[];repeats=[];history={};source_checks={}
    for model,batches in batches_by_model.items():
        reference=(REPO/next(x['source'] for x in old['paired_rows'] if x['baseline']==model)).with_name('predictions.csv')
        previous=csvrows(reference)
        for batch in batches:
            if recheck and f'{model}_batch{batch:02d}' not in expected:continue
            if selected_points and f'{model}_batch{batch:02d}' not in selected_points:continue
            folder=ROOT/f'{model}_batch{batch:02d}';cost=folder/'cost';util=folder/'util'
            cr,ur=read(cost/'report.json'),read(util/'report.json')
            assert cr['status']==ur['status']=='complete' and cr['test_videos']==558 and cr['cost_test_videos']==200
            assert cr['explicit_warmup_forwards']==0 and cr['first_test_batch_included'] is True
            assert cr['checkpoint_sha256']==checkpoints[model] and cr['manifest_sha256']==SHA
            guard(cost/'process_audit.json',cr['pid'])
            for stage in [cost,util]:
                receipt=read(stage/('recheck_driver_receipt.json' if recheck else 'selected_driver_receipt.json'))
                assert receipt['checkpoint_sha256']==checkpoints[model] and receipt['batch']==batch and receipt['baseline']==model
                for path,h in receipt['source_sha256'].items():
                    if suite:assert suite['source_sha256'].get(path)==h
                    elif model=='qwen':assert cr['source_sha256'].get(path)==h
                    key=(path,h)
                    if key not in source_checks:
                        content=subprocess.check_output(['git','show',cr['git_commit']+':'+path],cwd=REPO)
                        assert hashlib.sha256(content).hexdigest()==h;source_checks[key]=True
            predictions=csvrows(cost/'predictions.csv');timings=csvrows(cost/'batch_timing.csv');telemetry=csvrows(cost/'telemetry.csv')
            for p in [cost/'predictions.csv',cost/'batch_timing.csv',cost/'telemetry.csv',cost/'report.json',util/'report.json']:
                assert p.resolve() in verified,str(p)
            assert len(predictions)==len({x['sample_id'] for x in predictions})==558
            assert [(x['sample_id'],float(x['target_mos']),json.loads(x['selected_frame_positions'])) for x in predictions]==[(x['sample_id'],float(x['target_mos']),json.loads(x['selected_frame_positions'])) for x in previous]
            ys=[float(x['target_mos']) for x in predictions];ps=[float(x['prediction']) for x in predictions]
            srcc,plcc=pearson(ranks(ys),ranks(ps)),pearson(ys,ps)
            assert abs(srcc-cr['performance']['srcc'])<1e-12 and abs(plcc-cr['performance']['plcc'])<1e-12
            sizes=[min(batch,end-start) for begin,end in [(0,200),(200,558)] for start in range(begin,end,batch)]
            assert [int(x['videos']) for x in timings]==sizes and sum(sizes)==558
            first=[x for x in timings if x['in_cost_subset']=='True'];assert sum(int(x['videos']) for x in first)==200
            active=[x for x in telemetry if x['phase'].startswith('active:')];assert active
            watts=statistics.mean(float(x['watts']) for x in active);wall=sum(float(x['synchronized_wall_ms']) for x in first)
            t16=wall*16/200;e16=(338.2+watts)*t16/1000
            assert abs(watts-cr['power']['active_mean_w'])<1e-10
            assert abs(t16-cr['cost']['latency_16_videos_ms'])<1e-9 and abs(e16-cr['cost']['energy_16_videos_j'])<1e-9
            identity=read(util/'identity.json');assert identity['checkpoint_sha256']==checkpoints[model] and identity['manifest_sha256']==SHA
            if model=='qwen':
                point=util
                assert ur['full558_exact'] and ur['cold_cost_predictions_equal']
                eq=read(util/'before_equality.json');same_outputs(eq)
                assert eq['normalized_learned_work_equal'] and work(eq['actual_control_signatures'])==work(eq['actual_candidate_signatures'])
                same_outputs(read(util/'after_equality.json'))
                strict=csvrows(util/'full558_predictions.csv');assert len(strict)==558
                assert all(float(x['native_mos'])==float(x['candidate_mos'])==float(x['cost_mos'])==ps[i] for i,x in enumerate(strict))
                for evidence in read(util/'full558_equality.json')['batches']:same_outputs(evidence['equality'])
                for name in ['before_process_audit.json','after_process_audit.json','full558_process_audit.json']:guard(util/name,identity['pid'])
                util_value=ur['mean_utilization_percent']
            else:
                assert len(ur['points'])==1 and ur['points'][0]['batch']==batch
                point=util/f'batch{batch:02d}';point_report=read(point/'report.json');util_value=point_report['mean_utilization_percent']
            total=0;n=0;stamp=0;means=[];samples_total=0;duration=0;calls=0;peak=0
            for repeat in [1,2,3]:
                d=point/f'repeat{repeat:02d}';rr=read(d/'report.json');window=rr['measurement'];raw=csvrows(d/'telemetry.csv')
                selected=[x for x in raw if window['start_monotonic_s']<=float(x['host_monotonic_s'])<window['end_monotonic_s']]
                values=[float(x['utilization_percent']) for x in selected]
                assert len(values)==window['samples'] and window['duration_s']>=30 and rr['warmup']['duration_s']>=5
                assert abs(statistics.mean(values)-window['mean_gpu_utilization_percent'])<1e-9
                groups=read(d/'forward_groups.json');assert all(x['forwards']==8 for x in groups)
                assert sum(x['forwards'] for x in groups if x['phase']=='measured')==rr['measured_forward_invocations']
                guard(d/'process_audit.json',identity['pid'])
                clock=[x for x in csvrows(d/'clock_telemetry.csv') if window['start_monotonic_s']<=float(x['host_monotonic_s'])<window['end_monotonic_s']]
                assert clock and all(abs(float(x['power_limit_w'])-250)<.1 for x in clock)
                avg=lambda k:statistics.mean(float(x[k]) for x in clock)
                stamp=max(stamp,float(selected[-1]['host_time_unix']));mean=statistics.mean(values);means.append(mean)
                total+=sum(values);n+=len(values);duration+=window['duration_s'];calls+=rr['measured_forward_invocations'];peak=max(peak,max(values))
                repeats.append(dict(model=NAMES[model],baseline=model,batch=batch,repeat=repeat,mean_percent=mean,peak_percent=max(values),samples=len(values),duration_s=window['duration_s'],forwards=rr['measured_forward_invocations'],power_w=window['mean_board_power_w'],memory_mib=window['mean_memory_used_mib'],sm_clock_mhz=avg('sm_clock_mhz'),temperature_c=avg('temperature_c'),warmup_s=rr['warmup']['duration_s'],cpu_queue_wait_ms=(rr['cpu_measurement_end']['main_thread_schedstat'][1]-rr['cpu_measurement_start']['main_thread_schedstat'][1])/1e6,measured_local=datetime.fromtimestamp(float(selected[-1]['host_time_unix']),timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S'),source=(d/'report.json').relative_to(REPO).as_posix()))
            assert abs(total/n-util_value)<1e-9
            sample_srcc=pearson(ranks(ys[:200]),ranks(ps[:200]))
            row=dict(model=NAMES[model],baseline=model,batch_size=batch,srcc=srcc,plcc=plcc,first200_srcc=sample_srcc,
                time_16_videos_ms=t16,energy_16_videos_j=e16,board_mean_w=watts,cost_wall_sum_ms=wall,cost_calls=len(first),
                first_cost_call_ms=float(first[0]['synchronized_wall_ms']),first_call_fraction_percent=100*float(first[0]['synchronized_wall_ms'])/wall,
                wall_mean_ms=wall/len(first),wall_p95_ms=sorted(float(x['synchronized_wall_ms']) for x in first)[int(.95*(len(first)-1))],
                active_samples=len(active),gpu_util_peak_percent=max(float(x['utilization_percent']) for x in active),
                memory_mean_mib=statistics.mean(float(x['memory_used_mib']) for x in active),memory_peak_mib=max(float(x['memory_used_mib']) for x in active),
                gpu_util_mean_percent=statistics.mean(float(x['utilization_percent']) for x in active),
                independent_resident_util_mean_percent=total/n,independent_resident_util_peak_percent=peak,util_repeat_means=means,
                util_samples=n,util_duration_s=duration,util_forward_calls=calls,util_sum_percent_points=total,
                measured_local=datetime.fromtimestamp(stamp,timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S'),
                cost_measured_local=datetime.fromtimestamp(float(telemetry[-1]['host_time_unix']),timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S'),
                checkpoint_sha256=checkpoints[model],source=(cost/'report.json').relative_to(REPO).as_posix(),
                utilization_source=(point/'report.json').relative_to(REPO).as_posix(),
                execution='equivalence-validated optimized Qwen execution' if model=='qwen' else 'native',
                parameter_metadata={k:catalog[model][k] for k in ['model_total_parameters_billion','moe_active_parameters_billion','parameter_basis','parameter_source']})
            result.append(row)
    if selected_points:assert len(result)==len(selected_points) and len(repeats)==3*len(selected_points)
    else:assert suite['completed']==expected and len(result)==len(expected) and len(repeats)==3*len(expected)
    target=ROOT/('verified_points_'+'_'.join(selected_points)+'.json' if selected_points else 'verified_selected_execution.json')
    output=dict(status='verified_partial' if selected_points else 'complete',paired_rows=result,repeats=repeats,ours=old['ours'],historical_qwen_B1=anchor,
        raw_files_SHA_verified=len(verified),published_source_checks=len(source_checks),no_Qwen_B1_execution=True,
        utilization_and_cost_are_distinct_workloads=True,all_cost_videos=200,all_performance_videos=558)
    target.open('x',encoding='utf-8').write(json.dumps(output,indent=2)+'\n')
    if recheck:native_history_comparison(output)
    print(json.dumps(dict(status=output['status'],points=len(result),repeats=len(repeats),source_checks=len(source_checks),raw_SHA_files=len(verified))))
    return output


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--points',nargs='+');parser.add_argument('--yolo-clip-recheck',action='store_true');parser.add_argument('--yolo-clip-sequential',action='store_true');parser.add_argument('--rental-extension',action='store_true');parser.add_argument('--kimi-clean-extension',action='store_true')
    parser.add_argument('--busy50-service-extension',action='store_true')
    args=parser.parse_args()
    if args.busy50_service_extension:audit_busy50_service_extension()
    elif args.kimi_clean_extension:audit_kimi_clean_extension()
    elif args.rental_extension:audit_rental_extension()
    elif args.yolo_clip_sequential:audit_sequential()
    else:
        if args.yolo_clip_recheck:ROOT=TASK/'runs/simulation/lgvq_yolo_clip_recheck_a100_20261017'
        audit(args.points,args.yolo_clip_recheck)
