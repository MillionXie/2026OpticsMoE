"""Resume-safe transfer of compact amplitudes or canonical CCDs for one stage."""
import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import paramiko

STAGES = ('vision_router', 'vision_expert', 'vision_global',
          'language_router', 'language_expert', 'language_global')
SERVER_ROOT = '/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t08_abo_image_text_retrieval/runs/physical/alpha040_10cm_20260925'
BENCH_ROOT = 'E:/code/guest/2026OpticsMoE/ABO_T2I_10cm_alpha040_20260925/full_test'


def connect(host, port, user, password):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, port=port, username=user, password=password, timeout=15)
    return c


def copy_one(source, target, source_path, target_path):
    size = source.stat(source_path).st_size
    replace = False
    try:
        if target.stat(target_path).st_size == size:
            return
        replace = True
    except FileNotFoundError:
        pass
    temporary = target_path + '.uploading'
    with source.open(source_path, 'rb') as src, target.open(temporary, 'wb') as dst:
        while chunk := src.read(1024 * 1024):
            dst.write(chunk)
    if target.stat(temporary).st_size != size:
        raise RuntimeError(f'Incomplete transfer: {temporary}')
    if replace:
        target.remove(target_path)
    target.rename(temporary, target_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=STAGES, required=True)
    p.add_argument('--direction', choices=('ccd-up', 'amplitude-down'), required=True)
    p.add_argument('--server-password', required=True)
    p.add_argument('--bench-password', required=True)
    p.add_argument('--available-only', action='store_true',
                   help='During an active capture, skip CCDs not yet saved')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--run-name', choices=('full_test', 'finetune_train800'), default='full_test')
    a = p.parse_args()
    server = connect('202.120.62.181', 24096, 'guest3', a.server_password)
    bench = connect('1.tcp.vip.cpolar.top', 12705, 'PS', a.bench_password)
    try:
        s, b = server.open_sftp(), bench.open_sftp()
        index = STAGES.index(a.stage)
        server_base = SERVER_ROOT if a.run_name == 'full_test' else SERVER_ROOT + '/' + a.run_name
        bench_base = BENCH_ROOT if a.run_name == 'full_test' else BENCH_ROOT.rsplit('/', 1)[0] + '/' + a.run_name
        server_stage = server_base + f'/{index+1:02d}_{a.stage}'
        bench_stage = bench_base + f'/{index+1:02d}_{a.stage}'
        if a.direction == 'ccd-up':
            src, dst = b, s
            src_dir, dst_dir = bench_stage + '/ccd_captured', server_stage + '/ccd_captured'
            names = sorted(name for name in src.listdir(src_dir) if name.endswith('.png'))
            expected = ((2400 if index < 3 else 2500) if a.run_name == 'full_test'
                        else (800 if index < 3 else 900))
            if len(names) != expected:
                raise RuntimeError(f'CCD count {len(names)} != {expected}')
        else:
            src, dst = s, b
            src_dir, dst_dir = server_stage + '/compact_amplitude', bench_stage + '/compact_amplitude'
            data = src.open(server_stage + '/manifest.jsonl').read()
            rows = list(map(json.loads, data.splitlines()))
            expected = ((2400 if index < 3 else 2500) if a.run_name == 'full_test'
                        else (800 if index < 3 else 900))
            if len({r['key'] for r in rows}) != expected:
                raise RuntimeError(f'Incomplete export: {len(rows)}/{expected}')
            names = [r['file'] for r in rows]
        try:
            dst.stat(dst_dir)
        except FileNotFoundError:
            parent = dst_dir.rsplit('/', 1)[0]
            try:
                dst.mkdir(parent)
            except OSError:
                pass
            dst.mkdir(dst_dir)
        if a.direction == 'amplitude-down':
            with dst.open(bench_stage + '/manifest.jsonl', 'wb') as f:
                f.write(data)
        if not 1 <= a.workers <= 8:
            raise ValueError('workers must be 1..8')
        local = threading.local()
        source_client = bench if a.direction == 'ccd-up' else server
        target_client = server if a.direction == 'ccd-up' else bench

        def one(name):
            if not hasattr(local, 'src'):
                local.src = source_client.open_sftp()
                local.dst = target_client.open_sftp()
            try:
                copy_one(local.src, local.dst, src_dir + '/' + name, dst_dir + '/' + name)
            except FileNotFoundError:
                if a.available_only and a.direction == 'ccd-up':
                    return
                raise

        with ThreadPoolExecutor(max_workers=a.workers) as executor:
            futures = [executor.submit(one, name) for name in names]
            for count, future in enumerate(as_completed(futures), 1):
                future.result()
                if count % 100 == 0:
                    print(f'{a.stage} {a.direction}: {count}/{expected}', flush=True)
        if not a.available_only:
            print(f'COMPLETE {a.stage} {a.direction}: {expected}/{expected}', flush=True)
    finally:
        server.close()
        bench.close()


if __name__ == '__main__':
    main()
