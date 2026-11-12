"""Run the declared zero-phase training stages through the existing trainer."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

if __package__:
    from .wait_for_gpu import idle_gpus
else:
    from wait_for_gpu import idle_gpus


ROOT = Path(__file__).resolve().parent


def write(path, record):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n', 'utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--train-soft-targets', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-wait-seconds', type=int, default=3600)
    args = parser.parse_args()
    profile = json.loads(args.config.read_text('utf-8'))
    if profile['base']['model']['phase_parameterization'] != 'direct':
        raise ValueError('This workflow requires exact-zero direct phase initialization')
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'configs').mkdir()
    (args.out / 'logs').mkdir()
    write(args.out / 'profile.json', profile)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    commit = os.environ.get('LIGHTGEN_SOURCE_COMMIT', commit)
    write(args.out / 'launch.json', {
        'command': sys.argv,
        'pid': os.getpid(),
        'source_commit': os.environ.get('LIGHTGEN_SOURCE_COMMIT', commit),
        'electronic_initialization': 'random',
        'phase_initialization': 'exact zero in both global stages',
        'fixed_front': 'extracted Qwen patch, position and prompt buffers',
    })
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible:
        if ',' in visible:
            raise ValueError('Select one GPU for this sequential workflow')
        visible = subprocess.check_output([
            'nvidia-smi', '-i', visible, '--query-gpu=uuid', '--format=csv,noheader',
        ], text=True).strip()
    completed = []
    parent = None
    for number, stage in enumerate(profile['stages']):
        config = copy.deepcopy(profile['base'])
        config.update(stage)
        name = config.pop('name')
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('Invalid stage name')
        if number == 0:
            config['initialization_snapshot'] = str(args.out / 'initialization.pt')
        config_path = args.out / 'configs' / (name + '.json')
        write(config_path, config)
        started = time.monotonic()
        while True:
            available = idle_gpus()
            eligible = [gpu for gpu in available if visible is None or gpu == visible]
            if eligible:
                visible = eligible[0]
                break
            if time.monotonic() - started > args.max_wait_seconds:
                raise TimeoutError('No idle GPU available for stage ' + name)
            write(args.out / 'status.json', {
                'state': 'waiting_for_idle_GPU', 'stage': name,
                'completed_stages': completed, 'pid': os.getpid(),
            })
            time.sleep(10)
        command = [sys.executable, str(ROOT / 'run.py'), '--config', str(config_path),
                   '--paths', str(args.paths.resolve()), '--out', str(args.out / name),
                   '--device', 'cuda', '--train-soft-targets', str(args.train_soft_targets.resolve())]
        if parent is not None:
            command.extend(['--initialize', str(parent)])
        environment = os.environ.copy()
        environment['CUDA_VISIBLE_DEVICES'] = visible
        with (args.out / 'logs' / (name + '.log')).open('w', buffering=1) as log:
            child = subprocess.Popen(command, cwd=ROOT, env=environment,
                                     stdout=log, stderr=subprocess.STDOUT)
            write(args.out / 'status.json', {
                'state': 'training', 'stage': name, 'training_pid': child.pid,
                'pid': os.getpid(), 'gpu_uuid': visible, 'completed_stages': completed,
            })
            code = child.wait()
        if code:
            write(args.out / 'status.json', {
                'state': 'failed', 'stage': name, 'returncode': code,
                'completed_stages': completed,
            })
            raise RuntimeError('Training failed in stage ' + name)
        parent = args.out / name / 'LightGen.pt'
        completed.append({
            'stage': name, 'checkpoint': str(parent),
            'metrics': json.loads((parent.parent / 'metrics.json').read_text('utf-8')),
        })
        write(args.out / 'stages.json', completed)
    write(args.out / 'status.json', {
        'state': 'completed', 'completed_stages': completed,
        'last_checkpoint': str(parent), 'source_commit': commit,
    })


if __name__ == '__main__':
    main()
