"""Resume the balanced TRAIN capture without modifying the original TEST set.

Stops on missing files, saturation, device failure, or an incomplete stage. The
program runs in the local orchestration computer; the SLM/camera run on the
bench and model generation runs on the server.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import paramiko

HERE = Path(__file__).resolve().parent
STAGES = ('vision_router', 'vision_expert', 'vision_global',
          'language_router', 'language_expert', 'language_global')
BENCH = 'E:/code/guest/2026OpticsMoE/ABO_T2I_10cm_alpha040_20260925/finetune_train800'
SERVER = ('/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/'
          't08_abo_image_text_retrieval/runs/physical/'
          'alpha040_10cm_20260925/finetune_train800')


def connect(host: str, port: int, user: str, password: str):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port=port, username=user, password=password, timeout=20)
    client.get_transport().set_keepalive(30)
    return client


def remote_text(client, path: str) -> str:
    try:
        with client.open_sftp() as sftp, sftp.open(path, 'rb') as stream:
            return stream.read().decode(errors='replace')
    except FileNotFoundError:
        return ''


def remote_count(client, path: str) -> int:
    try:
        with client.open_sftp() as sftp:
            return sum(name.lower().endswith('.png') for name in sftp.listdir(path))
    except FileNotFoundError:
        return 0


def invoke(script: str, *args: str):
    cmd = [sys.executable, str(HERE / script), *map(str, args)]
    print('RUN', script, ' '.join(arg for arg in map(str, args)
                                  if arg not in ('guest3', '100right')), flush=True)
    subprocess.run(cmd, cwd=HERE, check=True)


def wait_for(client, path: str, marker: str, timeout_s: int) -> None:
    deadline = time.monotonic() + timeout_s
    last = ''
    while time.monotonic() < deadline:
        body = remote_text(client, path)
        if marker in body:
            print('DONE', marker, flush=True)
            return
        if 'Traceback (most recent call last)' in body or 'CUDA out of memory' in body:
            raise RuntimeError(f'{path}:\n{body[-3000:]}')
        progress = [line for line in body.splitlines()
                    if line.startswith(('exported ', 'captured '))]
        if progress and progress[-1] != last:
            last = progress[-1]
            print(last, flush=True)
        time.sleep(15)
    raise TimeoutError(f'{marker} not found in {path}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--server-password', required=True)
    parser.add_argument('--bench-password', required=True)
    parser.add_argument('--from-stage', choices=STAGES, default='vision_router')
    parser.add_argument('--gpu', type=int, default=3)
    args = parser.parse_args()
    bench = connect('1.tcp.vip.cpolar.top', 12705, 'PS', args.bench_password)
    server = connect('202.120.62.181', 24096, 'guest3', args.server_password)
    try:
        for index in range(STAGES.index(args.from_stage), len(STAGES)):
            stage = STAGES[index]
            directory = f'{index + 1:02d}_{stage}'
            total = 800 if index < 3 else 900
            server_stage = f'{SERVER}/{directory}'
            bench_stage = f'{BENCH}/{directory}'
            export_log = f'{server_stage}/export.log'
            capture_log = f'{bench_stage}/capture.log'
            if f'{stage.upper()}_EXPORTED {total}/{total}' not in remote_text(server, export_log):
                invoke('launch_server_stage.py', '--stage', stage,
                       '--run-name', 'finetune_train800', '--gpu', str(args.gpu),
                       '--password', args.server_password)
            wait_for(server, export_log, f'{stage.upper()}_EXPORTED {total}/{total}', 3600)
            if remote_count(bench, f'{bench_stage}/compact_amplitude') != total:
                invoke('sync_full_stage.py', '--stage', stage,
                       '--run-name', 'finetune_train800', '--direction', 'amplitude-down',
                       '--server-password', args.server_password,
                       '--bench-password', args.bench_password, '--workers', '6')
            if remote_count(bench, f'{bench_stage}/ccd_captured') != total:
                body = remote_text(bench, capture_log)
                if not body or f'{stage.upper()}_COMPLETE {total}/{total}' not in body:
                    if index != 0 or remote_count(bench, f'{bench_stage}/ccd_captured') == 0:
                        invoke('launch_bench_stage.py', '--stage', stage,
                               '--run-name', 'finetune_train800',
                               '--exposure-us', '10000', '--wait-ms', '240',
                               '--password', args.bench_password)
                wait_for(bench, capture_log, f'{stage.upper()}_COMPLETE {total}/{total}', 3600)
            actual = remote_count(bench, f'{bench_stage}/ccd_captured')
            if actual != total:
                raise RuntimeError(f'{stage} bench CCD count {actual}/{total}')
            if remote_count(server, f'{server_stage}/ccd_captured') != total:
                invoke('sync_full_stage.py', '--stage', stage,
                       '--run-name', 'finetune_train800', '--direction', 'ccd-up',
                       '--server-password', args.server_password,
                       '--bench-password', args.bench_password, '--workers', '6')
            actual = remote_count(server, f'{server_stage}/ccd_captured')
            if actual != total:
                raise RuntimeError(f'{stage} server CCD count {actual}/{total}')
            print('STAGE VERIFIED', stage, f'{total}/{total}', flush=True)
        print('TRAIN800_ALL_SIX_STAGES_VERIFIED', flush=True)
    finally:
        bench.close()
        server.close()


if __name__ == '__main__':
    main()
