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


def audit(selected_points=None):
    suite=read(ROOT/'suite_status.json') if (ROOT/'suite_status.json').exists() else None
    if not selected_points:assert suite and suite['status']=='complete'
    cfg=read(TASK/'configs/baselines/selected_execution_20261016.json')
    expected=[f'{m}_batch{b:02d}' for m,bs in cfg['batches'].items() for b in bs]
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
    for model,batches in cfg['batches'].items():
        reference=(REPO/next(x['source'] for x in old['paired_rows'] if x['baseline']==model)).with_name('predictions.csv')
        previous=csvrows(reference)
        for batch in batches:
            if selected_points and f'{model}_batch{batch:02d}' not in selected_points:continue
            folder=ROOT/f'{model}_batch{batch:02d}';cost=folder/'cost';util=folder/'util'
            cr,ur=read(cost/'report.json'),read(util/'report.json')
            assert cr['status']==ur['status']=='complete' and cr['test_videos']==558 and cr['cost_test_videos']==200
            assert cr['checkpoint_sha256']==checkpoints[model] and cr['manifest_sha256']==SHA
            guard(cost/'process_audit.json',cr['pid'])
            for stage in [cost,util]:
                receipt=read(stage/'selected_driver_receipt.json')
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
    else:assert suite['completed']==expected and len(result)==16 and len(repeats)==48
    target=ROOT/('verified_points_'+'_'.join(selected_points)+'.json' if selected_points else 'verified_selected_execution.json')
    output=dict(status='verified_partial' if selected_points else 'complete',paired_rows=result,repeats=repeats,ours=old['ours'],historical_qwen_B1=anchor,
        raw_files_SHA_verified=len(verified),published_source_checks=len(source_checks),no_Qwen_B1_execution=True,
        utilization_and_cost_are_distinct_workloads=True,all_cost_videos=200,all_performance_videos=558)
    target.open('x',encoding='utf-8').write(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(status=output['status'],points=len(result),repeats=len(repeats),source_checks=len(source_checks),raw_SHA_files=len(verified))))
    return output


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--points',nargs='+')
    audit(parser.parse_args().points)
