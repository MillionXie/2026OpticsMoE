"""Strict sealed PT load and synthetic six-stage CPU replay, never dataset eval."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys


def run(repository, commit, checkpoint):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.dont_write_bytecode = True
    def blob(path):
        return subprocess.check_output(['git', '-C', str(repository), 'show', commit + ':' + path])
    namespace = {'__name__': 'git_tree_importer'}
    helper = 'maintenance/git_safety/check_t08_reverse_backend.py'
    exec(compile(blob(helper), helper, 'exec'), namespace)
    importer = namespace['GitImporter'](repository, commit, Path('/__git_memory_only__'))
    sys.meta_path.insert(0, importer)
    import torch
    torch.set_num_threads(2)
    expected = '25f23260864a00a0f32b5de209c27f5876166018c8d8ab0e880de81e0d2c2c22'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == expected
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model_module = importlib.import_module('LightGenV2.tasks.t07_abo_image_retrieval.standalone.model')
    replay = importlib.import_module('LightGenV2.tasks.t07_abo_image_retrieval.hardware.replay')
    model = model_module.OpticalRetrieval(payload['metadata'])
    model.load_state_dict(payload['state_dict'], strict=True)
    model.eval().requires_grad_(False)
    assert model.audit()['late_rgb_trainable_parameters'] == 92168
    generator = torch.Generator().manual_seed(72)
    batch = dict(input_ids=torch.tensor([model.metadata['template_ids']]),
                 image_grid_thw=torch.tensor([[1, 14, 14]]),
                 pixel_values=torch.randn(196, 1536, generator=generator))
    batch['attention_mask'] = torch.ones_like(batch['input_ids'])
    with torch.inference_mode():
        reference = replay.snapshot_simulation(model, batch)
    calls = []
    def ideal(stage, active, ids):
        assert stage == replay.STAGES[len(calls)]
        assert active.shape == (1, 478, 478)
        assert active.min() >= 0 and active.max() <= 1
        assert ids == ['synthetic_no_dataset']
        calls.append(stage)
        return reference[stage].clone(), {'synthetic': True}, 1.0
    result = replay.replay_batch(model, batch, ['synthetic_no_dataset'], ideal)
    difference = float((result['descriptor'] - reference['descriptor']).abs().max())
    assert calls == list(replay.STAGES)
    assert difference < 1e-5, difference
    assert all(value[0] > .999999 for value in result['stage_pcc'].values())
    # A second batch must clear old measured-router injections before simulation.
    calls.clear()
    second = replay.replay_batch(model, batch, ['synthetic_no_dataset'], ideal)
    assert torch.equal(result['descriptor'], second['descriptor'])
    for ids in ([], ['duplicate', 'duplicate']):
        try:
            replay.replay_batch(model, batch, ids, ideal)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid identities accepted')
    assert not any(name in sys.modules for name in ('four_image_flow', 'lab_dvp', 'shs_physical2400'))
    sys.meta_path.remove(importer)
    return dict(commit=commit, checkpoint_sha256=expected, strict_load=True,
                synthetic_samples=1, six_stage_calls=calls,
                maximum_descriptor_error=difference, repeated_replay_bit_identical=True,
                invalid_identity_checks=2, legacy_device_modules_imported=False,
                datasets_evaluated=False, hardware_touched=False, training=False,
                imported_source=importer.loaded)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.repository, args.commit, args.checkpoint), indent=2))
