"""Repeat complete inference processes under one recorded runtime protocol."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
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
    names = ('infer.py', 'model.py', 'visualize.py', 'weights/LightGen2.pt', 'data/test_rgb.npz')
    identities = {name: sha256(args.project / name) for name in names}
    expected_predictions = sha256(args.project / 'results/01_predictions.csv')
    protocol = {
        'source_commit': environment.get('LIGHTGEN_SOURCE_COMMIT'),
        'hardware': hardware, 'gpu_uuid': args.gpu_uuid,
        'cpu_affinity': sorted(os.sched_getaffinity(0)), 'threads': 4,
        'warmup_processes': args.warmups, 'measured_processes': args.repeats,
        'input_and_source_sha256': identities,
        'expected_predictions_sha256': expected_predictions,
        'program_scope': 'Python imports, weight and NPZ loading, all 558 predictions, metrics and figure exports, final CUDA synchronization',
        'process_scope': 'Interpreter launch through successful process exit',
        'cache_condition': 'One complete untimed invocation before measured fresh processes',
        'visualized_field': 1,
        'zip_extraction_during_measurements': False,
        'parent_python': platform.python_version(),
    }
    write_json(args.out / 'protocol.json', protocol)
    rows = []
    for index in range(args.warmups + args.repeats):
        phase = 'warmup' if index < args.warmups else 'measured'
        number = index + 1 if phase == 'warmup' else index - args.warmups + 1
        name = f'{phase}_{number:02d}'
        output = args.out / name
        command = [sys.executable, '-u', str(args.project / 'infer.py'), '--device', 'cuda',
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
                subprocess.run(command, cwd=args.project, env=environment,
                               stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
                process_seconds = time.perf_counter() - started
        finally:
            monitor.terminate()
            monitor.wait(timeout=10)
            telemetry.close()
        metrics = json.loads((output / '02_metrics.json').read_text('utf-8'))
        if metrics['videos'] != 558 or metrics['fields'] != 35 or metrics['device_name'] != hardware['device']:
            raise ValueError('Runtime sample or device identity differs')
        if sha256(output / '01_predictions.csv') != expected_predictions:
            raise ValueError('Repeated full evaluation predictions differ')
        row = {'phase': phase, 'repeat': number,
               'program_seconds': metrics['total_runtime_seconds'], 'process_seconds': process_seconds,
               'SRCC': metrics['SRCC'], 'load_average_before': list(load),
               'result_directory': name}
        rows.append(row)
        write_json(args.out / 'measurements.json', rows)
        write_json(args.out / 'status.json', {'state': 'running', 'completed_processes': len(rows)})
        print(json.dumps(row), flush=True)
    for name, expected in identities.items():
        if sha256(args.project / name) != expected:
            raise ValueError('Source or input changed during repeated measurements')
    measured = [row for row in rows if row['phase'] == 'measured']
    summary = {'hardware': hardware, 'measured_processes': len(measured),
               'videos_per_process': 558, 'predictions_bitwise_equal': True}
    for column in ('program_seconds', 'process_seconds'):
        values = [row[column] for row in measured]
        summary[column] = {'median': statistics.median(values), 'mean': statistics.mean(values),
                           'sample_standard_deviation': statistics.stdev(values),
                           'minimum': min(values), 'maximum': max(values)}
    median = summary['program_seconds']['median']
    summary['representative_result_directory'] = min(
        measured, key=lambda row: abs(row['program_seconds'] - median))['result_directory']
    write_json(args.out / 'summary.json', summary)
    with (args.out / 'measurements.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.out / 'status.json', {'state': 'completed'})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
