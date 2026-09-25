"""Copy verified final outputs locally and summarize six-stage capture QA."""
import argparse
import json
import statistics
from pathlib import Path

import paramiko

HERE = Path(__file__).resolve().parent
STAGES = ('vision_router', 'vision_expert', 'vision_global',
          'language_router', 'language_expert', 'language_global')
SERVER = '/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t08_abo_image_text_retrieval/runs/physical/alpha040_10cm_20260925'
BENCH = 'E:/code/guest/2026OpticsMoE/ABO_T2I_10cm_alpha040_20260925/full_test'


def connect(host, port, user, password):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, port=port, username=user, password=password, timeout=15)
    return c


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--server-password', required=True)
    p.add_argument('--bench-password', required=True)
    a = p.parse_args()
    server = connect('202.120.62.181', 24096, 'guest3', a.server_password)
    bench = connect('1.tcp.vip.cpolar.top', 12705, 'PS', a.bench_password)
    try:
        s, b = server.open_sftp(), bench.open_sftp()
        for name in ('physical_test_report.json', 'physical_test_predictions.csv'):
            s.get(SERVER + '/' + name, str(HERE / name))
            b.put(str(HERE / name), BENCH + '/' + name)
        summary = {'checkpoint_sha256': json.loads((HERE / 'physical_test_report.json').read_text())['checkpoint_sha256'],
                   'stages': []}
        for index, stage in enumerate(STAGES):
            directory = f'{index+1:02d}_{stage}'
            expected = 2400 if index < 3 else 2500
            bench_dir = BENCH + '/' + directory
            server_dir = SERVER + '/' + directory
            bench_files = [x for x in b.listdir(bench_dir + '/ccd_captured') if x.endswith('.png')]
            server_files = [x for x in s.listdir(server_dir + '/ccd_captured') if x.endswith('.png')]
            if len(bench_files) != expected or len(server_files) != expected:
                raise RuntimeError(f'{stage}: bench/server {len(bench_files)}/{len(server_files)}, expected {expected}')
            rows = [json.loads(line) for line in b.open(bench_dir + '/capture_journal.jsonl').read().decode().splitlines()]
            if len(rows) != expected:
                raise RuntimeError(f'{stage}: journal has {len(rows)}, expected {expected}')
            keys = {row['key'] for row in rows} if index > 0 else {f'image_{int(row["index"]):04d}' for row in rows}
            if len(keys) != expected:
                raise RuntimeError(f'{stage}: duplicate journal key')
            if index >= 3 and sum(key.startswith('title_') for key in keys) != 100:
                raise RuntimeError(f'{stage}: missing title CCDs')
            phase_hashes = {row['phase_sha256'] for row in rows}
            if len(phase_hashes) != 1:
                raise RuntimeError(f'{stage}: phase changed within stage')
            p99 = [float(row['p99']) for row in rows]
            stage_info = {'stage': stage, 'expected': expected,
                          'bench_ccd_count': len(bench_files),
                          'server_ccd_count': len(server_files),
                          'title_count': sum(key.startswith('title_') for key in keys),
                          'phase_sha256': next(iter(phase_hashes)),
                          'p99_min': min(p99), 'p99_median': statistics.median(p99),
                          'p99_max': max(p99),
                          'maximum_saturation_fraction': max(float(row['saturation_fraction']) for row in rows),
                          'exposure_us': sorted({row['exposure_us'] for row in rows}),
                          'wait_ms': sorted({row['wait_ms'] for row in rows})}
            summary['stages'].append(stage_info)
        (HERE / 'physical_capture_qa.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        b.put(str(HERE / 'physical_capture_qa.json'), BENCH + '/physical_capture_qa.json')
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        server.close()
        bench.close()


if __name__ == '__main__':
    main()
