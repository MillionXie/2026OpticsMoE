"""Upload and launch the resumable TEST router exporter on the training server."""
import argparse
from pathlib import Path

import paramiko

HERE = Path(__file__).resolve().parent
REMOTE = '/DATA/DATA1/guest3/2026OpticsMoE/.codex_tmp/t08_full_test_20260925'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--host', required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--user', required=True)
    p.add_argument('--password', required=True)
    a = p.parse_args()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(a.host, port=a.port, username=a.user, password=a.password, timeout=12)
    try:
        _, stdout, stderr = client.exec_command(f'mkdir -p {REMOTE}')
        if stdout.channel.recv_exit_status():
            raise RuntimeError(stderr.read().decode(errors='replace'))
        sftp = client.open_sftp()
        for name in ('export_router_pilot.py', 'export_full_vision_router.py', 'run_full_vision_router_server.sh'):
            sftp.put(str(HERE / name), f'{REMOTE}/{name}')
        sftp.close()
        command = (f'cd {REMOTE} && chmod +x run_full_vision_router_server.sh && '
                   'nohup ./run_full_vision_router_server.sh > export.log 2>&1 < /dev/null & echo $!')
        _, stdout, stderr = client.exec_command(command)
        print('remote_pid:', stdout.read().decode(errors='replace').strip())
        print(stderr.read().decode(errors='replace').strip())
    finally:
        client.close()


if __name__ == '__main__':
    main()
