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
from math import sqrt
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
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--points',nargs='+');parser.add_argument('--yolo-clip-recheck',action='store_true');parser.add_argument('--yolo-clip-sequential',action='store_true')
    args=parser.parse_args()
    if args.yolo_clip_sequential:audit_sequential()
    else:
        if args.yolo_clip_recheck:ROOT=TASK/'runs/simulation/lgvq_yolo_clip_recheck_a100_20261017'
        audit(args.points,args.yolo_clip_recheck)
