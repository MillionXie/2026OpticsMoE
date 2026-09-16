"""Fresh-process reload of the common electronic checkpoint and each optical checkpoint."""
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from model import Electronic, FrozenFusion
spec = importlib.util.spec_from_file_location('frozen_electronic_runner', Path(__file__).with_name('run.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
evaluate, save, sha, tensors_sha = runner.evaluate, runner.save, runner.sha, runner.tensors_sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False
    metadata = json.loads((args.run/'metadata.json').read_text())
    save(args.out/'metadata.json', dict(config=metadata['config'], command=sys.argv,
         git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
         training_git_commit=metadata['git_commit'], python=sys.version, torch=torch.__version__,
         gpu=torch.cuda.get_device_name(), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
         data_sha256=sha(args.data), source_run=str(args.run), operation='read-only checkpoint reevaluation'))
    assert sha(args.data) == metadata['data_sha256']
    with np.load(args.data, allow_pickle=False) as arrays:
        val = tuple(torch.from_numpy(arrays['validation_'+k].copy()) for k in ['images', 'labels', 'domains'])
        ids = arrays['validation_ids'].copy()
    frozen = json.loads((args.run/'frozen_electronic.json').read_text())
    path = args.run/'electronic/best_checkpoint.pt'
    assert sha(path) == frozen['checkpoint_sha256']
    electronic = Electronic(metadata['train_channel_mean'], metadata['train_channel_std']).cuda()
    electronic.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model'])
    assert tensors_sha(electronic.state_dict()) == frozen['tensors_sha256']
    reports = []
    for stage in metadata['config']['architectures']:
        summary = json.loads((args.run/stage/'summary.json').read_text())
        path = args.run/stage/'best_checkpoint.pt'
        assert sha(path) == summary['checkpoint_sha256']
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        assert checkpoint['frozen_electronic_sha256'] == frozen['tensors_sha256']
        model = FrozenFusion(electronic, stage, metadata['optical_config']).cuda()
        model.optical.load_state_dict(checkpoint['model'])
        metrics, probabilities = evaluate(model, val, metadata['config']['batch_size'])
        assert metrics == summary['validation']
        for branch, prob in probabilities.items():
            with (args.run/stage/('validation_'+branch+'.csv')).open() as f:
                rows = list(csv.DictReader(f))
            assert [r['sample_id'] for r in rows] == ids.tolist()
            recorded = np.array([[float(r[f'p{i}']) for i in range(10)] for r in rows], dtype=np.float32)
            assert np.array_equal(recorded, prob)
        assert tensors_sha(electronic.state_dict()) == frozen['tensors_sha256']
        reports.append(dict(stage=stage, predictions_bitwise_identical=True, metrics=metrics,
                            optical_checkpoint_sha256=sha(path), electronic_checkpoint_sha256=frozen['checkpoint_sha256']))
        del model; torch.cuda.empty_cache()
    save(args.out/'result.json', dict(passed=True, samples_per_model=len(ids), reports=reports,
                                    verifier_sha256=sha(Path(__file__))))
    save(args.out/'status.json', dict(state='complete'))
    print(json.dumps(dict(passed=True, samples_per_model=len(ids), models=len(reports))))


if __name__ == '__main__':
    main()
