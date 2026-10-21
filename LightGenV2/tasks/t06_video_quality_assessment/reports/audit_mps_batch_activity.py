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


def audit(root):
    root = Path(root)
    c = experiment.configuration()
    state = (json.loads((root/'suite_status.json').read_text()) if (root/'suite_status.json').exists()
             else dict(status='partial_local_backup',selections={}))
    points = []
    reports = {}
    verified_sources = set()
    for path in sorted(root.glob('*/report.json')):
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
            for batch_report in report['points']:
                batch = batch_report['batch']
                bf = folder/f'batch{batch:02d}'
                inputs = json.loads((bf/'inputs.json').read_text())
                assert len(inputs['sample_ids']) == batch and len(set(inputs['sample_ids'])) == batch
                for index,repeat in enumerate(batch_report['repeats'],1):
                    rf = bf/f'repeat{index:02d}'
                    rows = list(csv.DictReader((rf/'telemetry.csv').open(newline='')))
                    a,b = repeat['measured_start_monotonic_s'],repeat['measured_end_monotonic_s']
                    assert b-a >= 30
                    values = [float(r['utilization_percent']) for r in rows if a <= float(r['host_monotonic_s']) < b]
                    assert len(values) >= 100
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
        assert report['visible_multiprocessors'] == (108 if report['cap'] == 100 else 54)
        assert report['profiler_overhead_present'] == (report['mode'] == 'pilot')
        point = dict(model=report['model'],batch=report['batch'],cap=report['cap'],
            mode=report['mode'],repeat=report['repeat'],report=path.relative_to(root).as_posix(),
            report_sha256=experiment.original.frozen.sha256(path),
            calls_sha256=experiment.original.frozen.sha256(folder/'calls.json'),
            predictions_sha256=experiment.original.frozen.sha256(folder/'predictions.csv'),
            checkpoint_sha256=report['checkpoint_sha256'],manifest_sha256=report['manifest_sha256'],
            instrumented_activity_not_formal_NVML=True)
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
                time16_ms=sum(p['time16_ms'] for p in group)/len(group) if group else None,
                energy16_j=sum(p['energy16_j'] for p in group)/len(group) if group else None,
                srcc_repeats=[p['performance'].get('srcc') for p in group])
        selections[model] = dict(batch=chosen,caps=caps)
    return dict(status='passed',suite_status=state['status'],mps_released=state.get('mps_released'),
        audited_points=len(points),selections=selections,points=points,
        profile_metric_not_uninstrumented_utilization=True,no_paper_export_updates=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=experiment.ROOT)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = audit(args.root)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k != 'points'},indent=2))


if __name__ == '__main__':
    main()
