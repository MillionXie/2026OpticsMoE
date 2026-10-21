"""CPU-only independent replay of bounded conditional-activity batch experiments."""
import argparse
import csv
import json
import math
import statistics
import subprocess
import hashlib
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, kendalltau, pearsonr
from .. import measure_mps_batch_activity as experiment
from .. import tf_activity_clock as clock


def metrics(target,prediction):
    def statistic(function):
        result = function(target,prediction)
        return float(result.statistic if hasattr(result,'statistic') else result[0])
    difference = prediction-target
    return dict(srcc=statistic(spearmanr),krcc=statistic(kendalltau),plcc=statistic(pearsonr),
        rmse=float(np.sqrt(np.mean(np.square(difference)))),mae=float(np.mean(np.abs(difference))))


def historical_large_models():
    """Do not confuse earlier 80GB resident rates with new laboratory runs.

    Reuse ONLY independently audited distinct-video cold costs. The old resident
    time/energy fields are deliberately ignored. Utilization and costs have
    different physical GPU UUIDs and dates, disclosed as separate references.
    """
    from .audit_mps_dataset_once import audit_point
    base = experiment.TASK/'runs/simulation'
    utilization = base/'lgvq_busy_resource_20261020'
    cost_root = base/'lgvq_mps_dataset_once_rental_20261021_replacement02'
    values = []
    source_cache = {}
    for model in ('deepseek','kimi'):
        for cap in (100,50):
            folder = utilization/f'mps_{model}_native_{cap}'
            identity = json.loads((folder/'identity.json').read_text())
            for name,digest in identity['sources'].items():
                key = (identity['git_commit'],name)
                if key not in source_cache:
                    source_cache[key] = hashlib.sha256(subprocess.check_output(
                        ['git','show',key[0]+':'+name],cwd=experiment.REPO)).hexdigest()
                assert source_cache[key] == digest
            source = folder/f'mps_cap{cap}'
            report = json.loads((source/'report.json').read_text())
            raw = list(csv.DictReader((source/'telemetry.csv').open(newline='')))
            assert report['status']=='complete' and report['batch']==report['unique_videos']==1
            assert identity['model_floating_dtypes']==['torch.bfloat16']
            assert identity['visible_multiprocessors']==(108 if cap==100 else 54)
            means,counts = [],[]
            for repeat in report['repeats']:
                a,b = repeat['first_tf_start'],repeat['final_gpu_output']
                assert b-a >= 30 and repeat['unique_videos']==1
                samples = [float(x['utilization_percent']) for x in raw if a<=float(x['host_monotonic_s'])<b]
                mean = statistics.mean(samples)
                assert math.isclose(mean,repeat['measurement']['mean_gpu_utilization_percent'],abs_tol=1e-9)
                assert len(samples)==repeat['measurement']['samples']
                means.append(mean)
                counts.append(len(samples))
            assert len(means)==3 and min(means)>=50
            costs = []
            for repeat in (1,2,3):
                point = cost_root/f'{model}_cap{cap}_repeat{repeat:02d}'
                bound = json.loads((point/'identity.json').read_text())
                assert bound['checkpoint_sha256']==identity['checkpoint_sha256']
                assert bound['assets']==identity['assets']
                assert bound['floating_parameter_dtypes']==identity['model_floating_dtypes']
                assert bound['power_limit_w']==400 and '80GB' in bound['gpu_name']
                result = audit_point(point)
                costs.append(result)
                first = next(csv.DictReader((point/'predictions.csv').open(newline='')))
                assert identity['sample_ids']==[first['sample_id']]
                assert identity['frame_positions'][0]==json.loads(first['frame_positions'])
            values.append(dict(model=model,batch=1,cap=cap,provenance='historical_80GB_archive_not_new_measurement',
                resident_busy_repeats_percent=means,resident_mean_busy_percent=sum(x*n for x,n in zip(means,counts))/sum(counts),
                resident_source=(source/'report.json').relative_to(experiment.REPO).as_posix(),
                resident_report_sha256=experiment.original.frozen.sha256(source/'report.json'),
                resident_gpu_uuid=identity['gpu_uuid'],cost_gpu_uuid=costs[0]['gpu_uuid'],
                utilization_and_cost_separate_instances=identity['gpu_uuid']!=costs[0]['gpu_uuid'],
                checkpoint_sha256=identity['checkpoint_sha256'],
                time16_ms=statistics.mean(r['time16_ms'] for r in costs),
                energy16_j=statistics.mean(r['energy16_j'] for r in costs),
                srcc_repeats=[r['srcc'] for r in costs],costs=costs,
                ignore_old_resident_time_and_energy=True,no_new_rental=True))
    return dict(status='passed',values=values,
        historical_utilization_reference_not_current_same_device_certification=True,
        audit_source_sha256=experiment.original.frozen.sha256(Path(__file__)),
        no_new_GPU_execution=True,no_paper_export_updates=True)


def historical_glm_costs():
    """Audit available old costs, without adopting them as fresh measurements."""
    from .audit_mps_dataset_once import audit_point
    root = experiment.TASK/'runs/simulation/lgvq_mps_dataset_once_a100_20261020'
    values = []
    for cap in (100,50):
        group = [audit_point((root/f'glm_cap{cap}_repeat{r:02d}').resolve()) for r in (1,2,3)]
        assert all(p['gpu_uuid']==experiment.original.frozen.A100_UUID for p in group)
        values.append(dict(model='glm',batch=1,cap=cap,costs=group,
            time16_ms=statistics.mean(p['time16_ms'] for p in group),
            energy16_j=statistics.mean(p['energy16_j'] for p in group),
            provenance='available_historical_cost_not_fresh_or_automatically_adopted'))
    return dict(status='passed',values=values,no_new_GPU_execution=True,
        audit_source_sha256=experiment.original.frozen.sha256(Path(__file__)),
        no_paper_export_updates=True)


def audit(root):
    root = Path(root)
    c = experiment.configuration()
    manifest = experiment.TASK/'runs/simulation/lgvq_temporal_reproduction_a100_20261013/lgvq_train2250_test558.csv'
    if not manifest.exists():
        manifest = experiment.REPO/'experiments/lgvq_four_stage_optical_electronic_109_no_attention_vqa/artifacts/lgvq_train2250_test558.csv'
    assert experiment.original.frozen.sha256(manifest) == experiment.original.frozen.MANIFEST_SHA
    expected = [r for r in csv.DictReader(manifest.open(newline='')) if r['split']=='test']
    assert len(expected) == 558
    state = (json.loads((root/'suite_status.json').read_text()) if (root/'suite_status.json').exists()
             else dict(status='partial_local_backup',selections={}))
    report_paths = sorted(root.glob('*/report.json'))
    if 'protocol' in state:
        c = state['protocol']
    elif report_paths:
        c = json.loads(report_paths[0].read_text())['protocol']
    assert (c['test_videos'],c['cost_videos'],c['repeats'],c['power_limit_w']) == (558,200,3,250)
    points = []
    reports = {}
    verified_sources = set()
    for path in report_paths:
        report = json.loads(path.read_text())
        folder = path.parent
        source_key = (report['git_commit'],tuple(sorted(report['sources'].items())))
        if source_key not in verified_sources:
            for name,digest in report['sources'].items():
                blob = subprocess.check_output(['git','show',report['git_commit']+':'+name],cwd=experiment.REPO)
                assert hashlib.sha256(blob).hexdigest() == digest
            verified_sources.add(source_key)
        if report['mode'] == 'resident':
            assert report['status'] == 'complete' and report['no_formal_cost']
            assert report['visible_multiprocessors'] == (108 if report['cap'] == 100 else 54)
            selected = next((r['batch'] for r in report['points'] if r['minimum50_reached']),None)
            assert selected == report['selected_batch']
            observed_batches = [r['batch'] for r in report['points']]
            if report['cap'] == 100:
                assert observed_batches == c['candidate_batches'][report['model']][:len(observed_batches)]
            else:
                assert len(observed_batches) == 1
            for batch_report in report['points']:
                batch = batch_report['batch']
                bf = folder/f'batch{batch:02d}'
                inputs = json.loads((bf/'inputs.json').read_text())
                assert inputs['sample_ids'] == [r['sample_id'] for r in expected[:batch]]
                for index,repeat in enumerate(batch_report['repeats'],1):
                    rf = bf/f'repeat{index:02d}'
                    rows = list(csv.DictReader((rf/'telemetry.csv').open(newline='')))
                    a,b = repeat['measured_start_monotonic_s'],repeat['measured_end_monotonic_s']
                    assert b-a >= 30
                    values = [float(r['utilization_percent']) for r in rows if a <= float(r['host_monotonic_s']) < b]
                    assert len(values) >= 100
                    assert len(values) == repeat['measurement']['samples']
                    mean = statistics.mean(values)
                    assert math.isclose(mean,repeat['measurement']['mean_gpu_utilization_percent'],abs_tol=1e-9)
                    groups = json.loads((rf/'forward_groups.json').read_text())
                    assert all(g['forwards'] == 8 for g in groups)
                    assert sum(g['forwards'] for g in groups if g['phase']=='measured') == repeat['measured_forward_invocations']
                    warm = [g for g in groups if g['phase']=='warmup']
                    assert warm[-1]['end_monotonic_s']-warm[0]['start_monotonic_s'] >= 5
                    assert max(abs(x-y) for x,y in zip(inputs['reference_mos'],repeat['output'])) <= .0002
                    points.append(dict(model=report['model'],batch=batch,cap=report['cap'],mode='resident',
                        repeat=index,busy_percent=mean,nvml_samples=len(values),measured_s=b-a,
                        report=(rf/'report.json').relative_to(root).as_posix(),
                        report_sha256=experiment.original.frozen.sha256(rf/'report.json'),
                        not_formal_cost=True,not_exact_per_forward_utilization=True))
                assert batch_report['minimum50_reached'] == experiment.resident_qualifies(batch_report['repeats'])
            reports[folder.name] = report
            continue
        calls = json.loads((folder/'calls.json').read_text())
        predictions = list(csv.DictReader((folder/'predictions.csv').open(newline='')))
        count = 32 if report['mode'] == 'pilot' else 558
        plan = experiment.batches(report['batch'],report['mode'] == 'pilot')
        assert report['status'] == 'complete' and report['explicit_warmup_forwards'] == 0
        assert [(x['sample_index'],x['sample_index']+x['videos']) for x in calls] == plan
        assert [i for x in calls for i in x['sample_indices']] == list(range(count))
        assert [x for call in calls for x in call['sample_ids']] == [r['sample_id'] for r in predictions]
        assert [int(r['sample_index']) for r in predictions] == list(range(count))
        assert [r['sample_id'] for r in predictions] == [r['sample_id'] for r in expected[:count]]
        assert [float(r['target_mos']) for r in predictions] == [float(r['temporal']) for r in expected[:count]]
        assert report['visible_multiprocessors'] == (108 if report['cap'] == 100 else 54)
        assert report['profiler_overhead_present'] == (report['mode'] == 'pilot')
        point = dict(model=report['model'],batch=report['batch'],cap=report['cap'],
            mode=report['mode'],repeat=report['repeat'],report=path.relative_to(root).as_posix(),
            report_sha256=experiment.original.frozen.sha256(path),
            calls_sha256=experiment.original.frozen.sha256(folder/'calls.json'),
            predictions_sha256=experiment.original.frozen.sha256(folder/'predictions.csv'),
            checkpoint_sha256=report['checkpoint_sha256'],manifest_sha256=report['manifest_sha256'],
            instrumented_activity_not_formal_NVML=report['mode']=='pilot')
        if report['mode'] == 'pilot':
            trace = json.loads((folder/'cuda_trace.json').read_text())
            anchors = json.loads((folder/'clock_anchors.json').read_text())
            errors = [clock.calibration_error(json.loads((folder/(name+'.json')).read_text()),
                json.loads((folder/(name+'_anchors.json')).read_text())) for name in ('clock_before','clock_after')]
            replay = experiment.activity.summarize_trace(trace,calls,anchors,max(errors)+10)
            for field in ('conditional_kernel_busy_percent','lower_percent','upper_percent','kernel_union_us','tf_to_output_us'):
                assert math.isclose(replay[field],report['measurement'][field],abs_tol=1e-7)
            point.update(busy_percent=replay['conditional_kernel_busy_percent'],
                lower_percent=replay['lower_percent'],trace_sha256=experiment.original.frozen.sha256(folder/'cuda_trace.json'))
        else:
            assert report['no_profiling_during_cost'] and report['cost_videos'] == 200
            assert not (folder/'cuda_trace.json').exists()
            samples = [experiment.original.telemetry.Sample(host_time_unix=float(r['host_time_unix']),
                host_monotonic_s=float(r['host_monotonic_s']),phase=r['phase'],watts=float(r['watts']),
                utilization_percent=float(r['utilization_percent']),memory_used_mib=float(r['memory_used_mib']))
                for r in csv.DictReader((folder/'telemetry.csv').open(newline=''))]
            charged = [x for x in calls if x['in_cost_subset']]
            assert sum(x['videos'] for x in charged) == 200
            intervals,compute,gap = experiment.original.charged_intervals(charged)
            result = experiment.original.energy.interval_energy(intervals,samples,c['host_proxy_w'])
            assert math.isclose(result['elapsed_s'],compute+gap,abs_tol=1e-7)
            assert math.isclose(report['cost']['time16_ms'],result['elapsed_s']*80,abs_tol=1e-6)
            assert math.isclose(report['cost']['energy16_j'],result['energy_j']*.08,abs_tol=1e-6)
            performance = metrics(np.array([float(x['target_mos']) for x in predictions]),
                np.array([float(x['prediction']) for x in predictions]))
            for field in performance:
                assert math.isclose(performance[field],report['performance'][field],abs_tol=1e-9)
            point.update(time16_ms=result['elapsed_s']*80,energy16_j=result['energy_j']*.08,
                performance=performance,compute_s=compute,nonpreprocessing_gap_s=gap,
                first_cold_batch_ms=calls[0]['legacy_host_ms'],
                later_cost_batch_ms_mean=statistics.mean(x['legacy_host_ms'] for x in charged[1:]),
                largest_cost_gaps_ms=sorted([(charged[i]['preprocess_start']-charged[i-1]['gpu_output'])*1000
                    for i in range(1,len(charged))],reverse=True)[:5],
                bookkeeping_included_in_gap=True,
                max_power_sample_age_s=result['max_power_sample_age_s'])
        points.append(point)
        reports[folder.name] = report
    selections = {}
    for model,selection in state.get('selections',{}).items():
        attempts = {b:[r for r in reports.values() if r['model'] == model and r.get('batch') == b
            and r['cap'] == 100 and r['mode'] == 'pilot'] for b in c['candidate_batches'][model]}
        resident = [r for r in reports.values() if r['model'] == model and r['mode']=='resident' and r['cap']==100]
        chosen = resident[0]['selected_batch'] if resident else experiment.selected_batch(attempts,c['candidate_batches'][model])
        if 'batch' not in selection:
            assert chosen is None
            selections[model] = selection
            continue
        assert chosen == selection['batch']
        costs = [p for p in points if p['model'] == model and p['mode'] == 'cost']
        if selection['status'] == 'paired_cost_complete':
            assert len(costs) == 6
        caps = {}
        for cap in (100,50):
            group = [p for p in costs if p['cap'] == cap]
            util = [p for p in points if p['model'] == model and p['batch'] == chosen and p['cap'] == cap and p['mode'] in ('pilot','resident')]
            caps[str(cap)] = dict(completed_cost_repeats=len(group),
                utilization_mode=util[0]['mode'] if util else None,
                busy_repeats_percent=[p['busy_percent'] for p in util],
                resident_nvml_mean_percent=(sum(p['busy_percent']*p['nvml_samples'] for p in util)/sum(p['nvml_samples'] for p in util)
                    if util and all(p['mode']=='resident' for p in util) else None),
                time16_ms=sum(p['time16_ms'] for p in group)/len(group) if group else None,
                energy16_j=sum(p['energy16_j'] for p in group)/len(group) if group else None,
                time16_ms_repeats=[p['time16_ms'] for p in group],
                energy16_j_repeats=[p['energy16_j'] for p in group],
                time16_ms_sample_sd=statistics.stdev(p['time16_ms'] for p in group) if len(group)>1 else None,
                srcc_repeats=[p['performance'].get('srcc') for p in group])
        selections[model] = dict(batch=chosen,caps=caps)
    return dict(status='passed',suite_status=state['status'],mps_released=state.get('mps_released'),
        audited_points=len(points),selections=selections,points=points,
        audit_source_sha256=experiment.original.frozen.sha256(Path(__file__)),
        manifest_sha256=experiment.original.frozen.sha256(manifest),
        profile_metric_not_uninstrumented_utilization=any(p['mode']=='pilot' for p in points),
        resident_utilization_separate_from_cost=True,no_paper_export_updates=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=experiment.ROOT)
    parser.add_argument('--historical80',action='store_true')
    parser.add_argument('--historical-glm',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.historical80 and args.historical_glm:
        parser.error('Select only one historical archive')
    result = (historical_large_models() if args.historical80 else
              historical_glm_costs() if args.historical_glm else audit(args.root))
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k != 'points'},indent=2))


if __name__ == '__main__':
    main()
