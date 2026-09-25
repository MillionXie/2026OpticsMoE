"""Launch final six-stage physical text-to-image evaluation on GPU 1."""
import argparse
from pathlib import Path

import paramiko

HERE = Path(__file__).resolve().parent
ROOT = '/DATA/DATA1/guest3/2026OpticsMoE'
WORKTREE = ROOT + '/.worktrees/t08_text_to_image_20260920'
TOOLS = ROOT + '/.codex_tmp/t08_full_test_20260925'
PHYSICAL = ROOT + '/LightGenV2/tasks/t08_abo_image_text_retrieval/runs/physical/alpha040_10cm_20260925'
CONFIG = WORKTREE + '/LightGenV2/tasks/t08_abo_image_text_retrieval/configs/optical_text_to_image_64_10cm_compact_e0p5.yaml'
CHECKPOINT = ROOT + '/LightGenV2/tasks/t08_abo_image_text_retrieval/runs/simulation/optical_text_to_image_64_10cm_compact_e0p5_seed42_20260925/best_checkpoint.pt'
DATA = ROOT + '/data/abo_easy100_dataset_20260906'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--password', required=True)
    p.add_argument('--gpu', type=int, default=1)
    a = p.parse_args()
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect('202.120.62.181', port=24096, username='guest3', password=a.password, timeout=15)
    try:
        sftp = c.open_sftp()
        for name in ('export_full_next_stage.py', 'evaluate_full_physical.py'):
            sftp.put(str(HERE / name), TOOLS + '/' + name)
        sftp.close()
        command = (f'cd {WORKTREE} && nohup env CUDA_VISIBLE_DEVICES={a.gpu} '
                   f'HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH={WORKTREE}:{TOOLS} '
                   f'/home/guest3/miniconda3/envs/xml/bin/python -u {TOOLS}/evaluate_full_physical.py '
                   f'--config {CONFIG} --checkpoint {CHECKPOINT} --data-root {DATA} '
                   f'--physical-root {PHYSICAL} --batch-size 4 '
                   f'> {PHYSICAL}/evaluation.log 2>&1 < /dev/null & echo $!')
        _, stdout, stderr = c.exec_command(command)
        print('remote_pid:', stdout.read().decode(errors='replace').strip())
        print(ascii(stderr.read().decode(errors='replace')))
    finally:
        c.close()


if __name__ == '__main__':
    main()
