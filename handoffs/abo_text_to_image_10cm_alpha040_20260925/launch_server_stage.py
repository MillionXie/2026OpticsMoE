"""Launch one measured-input T08 stage export on an idle lab-server GPU."""
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
    p.add_argument('--stage', required=True)
    p.add_argument('--password', required=True)
    p.add_argument('--gpu', type=int, default=1)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--title-only', action='store_true')
    a = p.parse_args()
    stages = ('vision_router', 'vision_expert', 'vision_global',
              'language_router', 'language_expert', 'language_global')
    if a.stage not in stages[1:]:
        raise ValueError(a.stage)
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect('202.120.62.181', port=24096, username='guest3', password=a.password, timeout=15)
    try:
        sftp = c.open_sftp()
        sftp.put(str(HERE / 'export_full_next_stage.py'), TOOLS + '/export_full_next_stage.py')
        sftp.close()
        stage_dir = PHYSICAL + f'/{stages.index(a.stage)+1:02d}_{a.stage}'
        cmd = (f'mkdir -p {stage_dir} && cd {WORKTREE} && '
               f'nohup env CUDA_VISIBLE_DEVICES={a.gpu} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 '
               f'PYTHONPATH={WORKTREE}:{TOOLS} '
               f'/home/guest3/miniconda3/envs/xml/bin/python -u {TOOLS}/export_full_next_stage.py '
               f'--stage {a.stage} --config {CONFIG} --checkpoint {CHECKPOINT} '
               f'--data-root {DATA} --physical-root {PHYSICAL} --batch-size 4 '
               f'--limit {a.limit} {"--title-only " if a.title_only else ""}'
               f'> {stage_dir}/export.log 2>&1 < /dev/null & echo $!')
        _, stdout, stderr = c.exec_command(cmd)
        print('remote_pid:', stdout.read().decode(errors='replace').strip())
        print(ascii(stderr.read().decode(errors='replace')))
    finally:
        c.close()


if __name__ == '__main__':
    main()
