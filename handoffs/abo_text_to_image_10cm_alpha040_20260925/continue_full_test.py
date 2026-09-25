"""Supervised, resumable continuation of the pinned six-stage physical TEST.

Starts after the active vision_expert capture. Each stage is audited for a
completion marker before its CCDs are used to produce the next stage. Low PCC
is never a rejection condition. Stops on hardware/file integrity errors.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import paramiko

HERE = Path(__file__).resolve().parent
STAGES = ('vision_router', 'vision_expert', 'vision_global',
          'language_router', 'language_expert', 'language_global')
BENCH = 'E:/code/guest/2026OpticsMoE/ABO_T2I_10cm_alpha040_20260925/full_test'
SERVER = '/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t08_abo_image_text_retrieval/runs/physical/alpha040_10cm_20260925'


def connect(host, port, user, password):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, port=port, username=user, password=password, timeout=20)
    c.get_transport().set_keepalive(30)
    return c


def read_remote(client, path):
    try:
        with client.open_sftp() as sftp:
            with sftp.open(path, 'rb') as stream:
                return stream.read().decode(errors='replace')
    except FileNotFoundError:
        return ''


def count_remote(client, path):
    try:
        with client.open_sftp() as sftp:
            return sum(name.endswith('.png') for name in sftp.listdir(path))
    except FileNotFoundError:
        return 0


def command(script, *args):
    cmd = [sys.executable, str(HERE / script), *map(str, args)]
    print('RUN', script, flush=True)
    result = subprocess.run(cmd, cwd=HERE, check=False)
    if result.returncode:
        raise RuntimeError(f'{script} failed with exit {result.returncode}')


def wait_log(client, path, marker, *, timeout_hours=4):
    until = time.monotonic() + timeout_hours * 3600
    previous = ''
    while time.monotonic() < until:
        body = read_remote(client, path)
        if marker in body:
            print('DONE', marker, flush=True)
            return
        if 'Traceback (most recent call last)' in body or 'CUDA out of memory' in body:
            raise RuntimeError(f'Failed log {path}:\n{body[-2500:]}')
        lines = [line for line in body.splitlines()
                 if line.startswith(('captured ', 'exported ', 'evaluated '))]
        if lines and lines[-1] != previous:
            previous = lines[-1]
            print(previous, flush=True)
        time.sleep(20)
    raise TimeoutError(f'No completion marker {marker}: {path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--server-password', required=True)
    p.add_argument('--bench-password', required=True)
    p.add_argument('--from-stage', choices=STAGES[1:], default='vision_expert')
    a = p.parse_args()
    bench = connect('1.tcp.vip.cpolar.top', 12705, 'PS', a.bench_password)
    server = connect('202.120.62.181', 24096, 'guest3', a.server_password)
    try:
        start = STAGES.index(a.from_stage)
        for index in range(start, len(STAGES)):
            stage = STAGES[index]
            directory = f'{index+1:02d}_{stage}'
            capture_log = f'{BENCH}/{directory}/capture.log'
            expected = 2400 if index < 3 else 2500
            if index > start:
                export_log = f'{SERVER}/{directory}/export.log'
                if f'{stage.upper()}_EXPORTED {expected}/{expected}' not in read_remote(server, export_log):
                    command('launch_server_stage.py', '--stage', stage,
                            '--password', a.server_password)
                    wait_log(server, export_log,
                             f'{stage.upper()}_EXPORTED {expected}/{expected}')
                transfer_log = (HERE / f'{stage}_amplitude_transfer.log').open('a', encoding='utf-8')
                try:
                    transfer = subprocess.Popen(
                        [sys.executable, str(HERE / 'sync_full_stage.py'), '--stage', stage,
                         '--direction', 'amplitude-down', '--server-password', a.server_password,
                         '--bench-password', a.bench_password],
                        cwd=HERE, stdout=transfer_log, stderr=subprocess.STDOUT,
                    )
                finally:
                    transfer_log.close()
                input_dir = f'{BENCH}/{directory}/compact_amplitude'
                until = time.monotonic() + 600
                while count_remote(bench, input_dir) < 100:
                    if transfer.poll() is not None:
                        raise RuntimeError(f'{stage} amplitude transfer stopped early')
                    if time.monotonic() > until:
                        raise TimeoutError(f'{stage} amplitudes not arriving')
                    time.sleep(10)
                command('launch_bench_stage.py', '--stage', stage,
                        '--password', a.bench_password, '--exposure-us', 10000,
                        '--wait-ms', 240)
            else:
                transfer = None
            wait_log(bench, capture_log, f'{stage.upper()}_COMPLETE {expected}/{expected}')
            actual = count_remote(bench, f'{BENCH}/{directory}/ccd_captured')
            if actual != expected:
                raise RuntimeError(f'{stage} captured {actual}/{expected} images')
            if transfer is not None:
                if transfer.wait(timeout=1800):
                    raise RuntimeError(f'{stage} amplitude transfer failed')
            command('sync_full_stage.py', '--stage', stage, '--direction', 'ccd-up',
                    '--server-password', a.server_password,
                    '--bench-password', a.bench_password)
            remote_count = count_remote(server, f'{SERVER}/{directory}/ccd_captured')
            if remote_count != expected:
                raise RuntimeError(f'{stage} server CCD count {remote_count}/{expected}')
            print(f'STAGE VERIFIED {stage} {expected}/{expected}', flush=True)
        command('launch_server_eval.py', '--password', a.server_password)
        wait_log(server, f'{SERVER}/evaluation.log',
                 '"all_six_optical_stages_measured": true', timeout_hours=4)
        report = json.loads(read_remote(server, f'{SERVER}/physical_test_report.json'))
        (HERE / 'physical_test_report.json').write_text(
            json.dumps(report, indent=2), encoding='utf-8'
        )
        print('FINAL PHYSICAL METRICS', json.dumps(report['metrics'], indent=2), flush=True)
    finally:
        bench.close()
        server.close()


if __name__ == '__main__':
    main()
