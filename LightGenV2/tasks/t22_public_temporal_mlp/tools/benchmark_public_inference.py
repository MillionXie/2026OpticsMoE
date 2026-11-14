"""Repeat complete inference processes under one recorded runtime protocol."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def compare_predictions(actual, reference, tolerance=1e-4):
    """Validate cross-device roundoff while preserving video and target identities."""
    with Path(actual).open(newline='') as stream:
        actual_rows = list(csv.DictReader(stream))
    with Path(reference).open(newline='') as stream:
        reference_rows = list(csv.DictReader(stream))
    if not actual_rows or len(actual_rows) != len(reference_rows):
        raise ValueError('Prediction sample counts differ')
    maximum_difference = 0.0
    for row, expected in zip(actual_rows, reference_rows):
        if row['sample_id'] != expected['sample_id'] or float(row['target_MOS']) != float(expected['target_MOS']):
            raise ValueError('Prediction video or target identities differ')
        value, reference_value = float(row['predicted_MOS']), float(expected['predicted_MOS'])
        if not math.isfinite(value) or not math.isfinite(reference_value):
            raise ValueError('Predictions must be finite')
        maximum_difference = max(maximum_difference, abs(value - reference_value))
    if maximum_difference > tolerance:
        raise ValueError('Predictions exceed the cross-device roundoff tolerance')
    return maximum_difference


def execution_plan(projects, warmups, repeats):
    for phase, count in (('warmup', warmups), ('measured', repeats)):
        for index in range(count):
            ordered = list(projects.items())
            if index % 2:
                ordered.reverse()
            for label, project in ordered:
                yield label, project, phase, index + 1


def summarize(rows):
    result = {}
    for column in ('program_seconds', 'process_seconds'):
        values = [row[column] for row in rows]
        result[column] = {'median': statistics.median(values), 'mean': statistics.mean(values),
                          'sample_standard_deviation': statistics.stdev(values),
                          'minimum': min(values), 'maximum': max(values)}
    median = result['program_seconds']['median']
    result['representative_result_directory'] = min(
        rows, key=lambda row: abs(row['program_seconds'] - median))['result_directory']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--comparison-project', type=Path,
                        help='Optional identical extracted project for alternating paired measurements')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--cpu-ids', help='Optional comma-separated Linux CPU affinity')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmups', type=int, default=1)
    args = parser.parse_args()
    if args.repeats < 3 or args.warmups < 1:
        raise ValueError('Use at least one full warmup and three measured processes')
    args.project = args.project.resolve()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    if args.cpu_ids:
        selected = {int(value) for value in args.cpu_ids.split(',')}
        if not selected.issubset(os.sched_getaffinity(0)):
            raise ValueError('Requested CPUs are outside the available affinity')
        os.sched_setaffinity(0, selected)
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                       OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', PYTHONPATH='')
    probe = "import json,platform,torch; print(json.dumps({'python':platform.python_version(),'torch':str(torch.__version__),'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0)}))"
    hardware = json.loads(subprocess.check_output([sys.executable, '-c', probe],
                                                  env=environment, text=True))
    names = ('infer.py', 'model.py', 'visualize.py', 'weights/LightGen2.pt',
             'data/input_for_review.npz', 'data/reference_mos.csv')
    identities = {name: sha256(args.project / name) for name in names}
    projects = {'reference': args.project}
    if args.comparison_project:
        projects['extracted'] = args.comparison_project.resolve()
        comparison = {name: sha256(projects['extracted'] / name) for name in names}
        if comparison != identities:
            raise ValueError('Comparison source, checkpoint or input differs')
    expected_predictions = sha256(args.project / 'results/01_predictions.csv')
    expected_srcc = json.loads((args.project / 'results/02_metrics.json').read_text('utf-8'))['SRCC']
    protocol = {
        'source_commit': environment.get('LIGHTGEN_SOURCE_COMMIT'),
        'hardware': hardware, 'gpu_uuid': args.gpu_uuid,
        'cpu_affinity': sorted(os.sched_getaffinity(0)), 'threads': 4,
        'warmup_processes': args.warmups, 'measured_processes': args.repeats,
        'input_and_source_sha256': identities,
        'expected_predictions_sha256': expected_predictions,
        'cross_device_MOS_absolute_tolerance': 1e-4,
        'cross_device_SRCC_absolute_tolerance': 1e-12,
        'program_scope': 'Python imports, weight and NPZ loading, all 558 predictions, metrics and figure exports, final CUDA synchronization',
        'process_scope': 'Interpreter launch through successful process exit',
        'cache_condition': 'One complete untimed invocation before measured fresh processes',
        'visualized_field': 1,
        'zip_extraction_during_measurements': False,
        'parent_python': platform.python_version(),
        'projects': {label: str(project) for label, project in projects.items()},
        'comparison_order': 'Alternate project order between repeated pairs',
    }
    write_json(args.out / 'protocol.json', protocol)
    rows = []
    device_predictions = None
    maximum_reference_difference = 0.0
    for label, project, phase, number in execution_plan(projects, args.warmups, args.repeats):
        name = f'{phase}_{number:02d}'
        if len(projects) > 1:
            name = label + '_' + name
        output = args.out / name
        command = [sys.executable, '-u', str(project / 'infer.py'), '--device', 'cuda',
                   '--visualize-field', '1', '--output', str(output)]
        telemetry = (args.out / (name + '_gpu.csv')).open('w')
        monitor = subprocess.Popen([
            'nvidia-smi', '--id=' + args.gpu_uuid,
            '--query-gpu=timestamp,uuid,utilization.gpu,memory.used,power.draw,clocks.sm,clocks.mem',
            '--format=csv,noheader,nounits', '--loop-ms=1000',
        ], stdout=telemetry, stderr=subprocess.STDOUT)
        load = os.getloadavg()
        try:
            with (args.out / (name + '.log')).open('w') as log:
                started = time.perf_counter()
                subprocess.run(command, cwd=project, env=environment,
                               stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
                process_seconds = time.perf_counter() - started
        finally:
            monitor.terminate()
            monitor.wait(timeout=10)
            telemetry.close()
        metrics = json.loads((output / '02_metrics.json').read_text('utf-8'))
        if metrics['videos'] != 558 or metrics['device_name'] != hardware['device']:
            raise ValueError('Runtime sample or device identity differs')
        actual_predictions = sha256(output / '01_predictions.csv')
        if device_predictions is None:
            maximum_reference_difference = compare_predictions(
                output / '01_predictions.csv', args.project / 'results/01_predictions.csv')
            if not math.isclose(metrics['SRCC'], expected_srcc, rel_tol=0, abs_tol=1e-12):
                raise ValueError('SRCC differs from the published reference')
            device_predictions = actual_predictions
            protocol['device_predictions_sha256'] = device_predictions
            protocol['maximum_MOS_difference_from_published_reference'] = maximum_reference_difference
            write_json(args.out / 'protocol.json', protocol)
        if actual_predictions != device_predictions:
            raise ValueError('Repeated same-device evaluation predictions differ')
        row = {'project': label, 'phase': phase, 'repeat': number,
               'program_seconds': metrics['total_runtime_seconds'], 'process_seconds': process_seconds,
               'SRCC': metrics['SRCC'], 'load_average_before': list(load),
               'result_directory': name}
        rows.append(row)
        write_json(args.out / 'measurements.json', rows)
        write_json(args.out / 'status.json', {'state': 'running', 'completed_processes': len(rows)})
        print(json.dumps(row), flush=True)
    for project in projects.values():
        for name, expected in identities.items():
            if sha256(project / name) != expected:
                raise ValueError('Source or input changed during repeated measurements')
    measured = [row for row in rows if row['phase'] == 'measured']
    per_project = {label: summarize([row for row in measured if row['project'] == label])
                   for label in projects}
    summary = {'hardware': hardware, 'measured_processes': args.repeats,
               'total_measured_processes': len(measured),
               'videos_per_process': 558, 'predictions_bitwise_equal': True}
    summary['maximum_MOS_difference_from_published_reference'] = maximum_reference_difference
    summary.update(per_project['reference'])
    summary['projects'] = per_project
    write_json(args.out / 'summary.json', summary)
    with (args.out / 'measurements.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.out / 'status.json', {'state': 'completed'})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
