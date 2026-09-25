"""Copy exact compact TEST router inputs from training server to bench PC."""
import argparse
import json
from pathlib import Path
import paramiko

SRC = '/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t08_abo_image_text_retrieval/runs/physical/alpha040_10cm_20260925/01_vision_router_export'
DST = 'E:/code/guest/2026OpticsMoE/ABO_T2I_10cm_alpha040_20260925/full_test/01_vision_router'


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
        src, dst = server.open_sftp(), bench.open_sftp()
        _, stdout, stderr = bench.exec_command('cmd /c mkdir E:\\code\\guest\\2026OpticsMoE\\ABO_T2I_10cm_alpha040_20260925\\full_test\\01_vision_router\\compact_amplitude')
        stdout.channel.recv_exit_status()
        manifest_bytes = src.open(SRC + '/manifest.jsonl', 'rb').read()
        rows = [json.loads(line) for line in manifest_bytes.splitlines()]
        by_index = {int(row['index']): row for row in rows}
        if len(by_index) != 2400:
            raise RuntimeError(f'Exporter incomplete: {len(by_index)}/2400')
        with dst.open(DST + '/manifest.jsonl', 'wb') as stream:
            stream.write(manifest_bytes)
        for i in range(2400):
            name = by_index[i]['file']
            source = SRC + '/compact_amplitude/' + name
            target = DST + '/compact_amplitude/' + name
            try:
                if dst.stat(target).st_size == src.stat(source).st_size:
                    continue
            except FileNotFoundError:
                pass
            with src.open(source, 'rb') as original, dst.open(target, 'wb') as output:
                while data := original.read(1024 * 1024):
                    output.write(data)
            if (i + 1) % 100 == 0:
                print(f'transferred {i + 1}/2400', flush=True)
        print('complete: 2400 compact inputs', flush=True)
    finally:
        server.close()
        bench.close()


if __name__ == '__main__':
    main()
