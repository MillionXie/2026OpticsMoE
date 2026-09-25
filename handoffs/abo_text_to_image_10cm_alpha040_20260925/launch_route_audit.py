"""Audit both measured optical routers after a completed physical TEST."""
import argparse
import json
from pathlib import Path

import paramiko

HERE = Path(__file__).resolve().parent
ROOT = '/DATA/DATA1/guest3/2026OpticsMoE'
WORKTREE = ROOT + '/.worktrees/t08_text_to_image_20260920'
TOOLS = ROOT + '/.codex_tmp/t08_full_test_20260925'
PHYSICAL = ROOT + '/LightGenV2/tasks/t08_abo_image_text_retrieval/runs/physical/alpha040_10cm_20260925'
CONFIG = WORKTREE + '/LightGenV2/tasks/t08_abo_image_text_retrieval/configs/optical_text_to_image_64_10cm_compact_e0p5.yaml'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--password', required=True)
    a = p.parse_args()
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect('202.120.62.181', port=24096, username='guest3', password=a.password, timeout=15)
    try:
        sftp = c.open_sftp()
        sftp.put(str(HERE / 'audit_physical_router.py'), TOOLS + '/audit_physical_router.py')
        for stage in ('vision_router', 'language_router'):
            cmd = (f'cd {WORKTREE} && PYTHONPATH={WORKTREE}:{TOOLS} '
                   f'/home/guest3/miniconda3/envs/xml/bin/python '
                   f'{TOOLS}/audit_physical_router.py --config {CONFIG} '
                   f'--physical-root {PHYSICAL} --stage {stage}')
            _, stdout, stderr = c.exec_command(cmd)
            out = stdout.read().decode(errors='replace')
            error = stderr.read().decode(errors='replace')
            if stdout.channel.recv_exit_status():
                raise RuntimeError(error[-2000:])
            name = stage + '_routing_summary.json'
            sftp.get(PHYSICAL + '/' + name, str(HERE / name))
            print(out, flush=True)
    finally:
        c.close()


if __name__ == '__main__':
    main()
