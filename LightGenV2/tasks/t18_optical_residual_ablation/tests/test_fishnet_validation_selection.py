"""Validation-only selection and the prepared-data identity boundary."""
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK))
import fishnet_selection as policy


def profiles():
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    return (read(TASK / 'fishnet_validation_profile.json'), read(TASK / 'fishnet_profile.json'),
            read(TASK / 'reports/fishnet_data_manifest_20261011.json'))


def test_selection_uses_loss_and_keeps_first_tie():
    assert policy.validation_improves({'balanced_nll': .3, 'accuracy': .5}, .4)
    assert not policy.validation_improves({'balanced_nll': .4, 'accuracy': 1.}, .3)
    assert not policy.validation_improves({'balanced_nll': .3, 'accuracy': 1.}, .3)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -.1])
def test_invalid_validation_loss_is_rejected(value):
    with pytest.raises(AssertionError):
        policy.validation_improves({'balanced_nll': value}, math.inf)


def test_new_selection_keeps_original_preparation_profile():
    profile, prepared, manifest = profiles()
    assert policy.selection_mode(profile) == 'validation'
    policy.validate_prepared_profile(profile, manifest, prepared, manifest['profile_sha256'])
    profile['lr'] *= 2
    with pytest.raises(AssertionError, match='contract'):
        policy.validate_prepared_profile(profile, manifest, prepared, manifest['profile_sha256'])


def test_wrong_preparation_identity_is_rejected():
    profile, prepared, manifest = profiles()
    with pytest.raises(AssertionError, match='identity'):
        policy.validate_prepared_profile(profile, manifest, prepared, '0' * 64)


@pytest.mark.skipif(sys.platform == 'win32', reason='Native Torch check runs on verified Linux; do not modify Windows DLL environment')
def test_training_never_reads_test_until_validation_checkpoint_is_locked(tmp_path, monkeypatch):
    import torch
    import fishnet_ablation as adapter

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.raw_phase = torch.nn.Parameter(torch.zeros(449848))

    profile, _, _ = profiles()
    profile = {**profile, 'epochs': 3}
    cfg = {key: profile[key] for key in ['epochs', 'batch_size', 'lr', 'ema_decay',
                                        'label_smoothing', 'augmentation', 'capture_weight', 'phase_smooth_weight', 'scope']}
    cfg.update(microbatch=2, evaluation_batch_size=2, selection_split='val')
    data_path = tmp_path / 'data.npz'
    data_path.write_bytes(b'identity fixture')
    manifest = {'cache_sha256': adapter.r.sha(data_path)}
    a = SimpleNamespace(data=data_path, out=tmp_path / 'arm', depth=2, rho=0.)
    a.out.mkdir()
    loaded, test_calls, val_calls = [], [], []
    samples = {split: (torch.ones(7, 1), torch.arange(7), split) for split in ['train', 'val', 'test']}

    def load_data(path, split):
        loaded.append(split)
        if split == 'test':
            lock = json.loads((a.out / 'selection_lock.json').read_text())
            assert lock['epoch'] == 1 and lock['test_evaluations_before_lock'] == 0
            assert lock['checkpoint_sha256'] == adapter.r.sha(a.out / 'moe_L2_seed17/best_checkpoint.pt')
        return samples[split]

    def forward(model, x, arch):
        logits = torch.stack([model.raw_phase.mean(), *[model.raw_phase.mean() * 0 for _ in range(7)]])
        return logits.softmax(0).expand(len(x), 8), torch.ones(len(x)), None

    def evaluate(model, examples, arch, batch):
        if examples[2] == 'test':
            assert loaded[-1] == 'test' and len(val_calls) == 5
            test_calls.append('once')
        if examples[2] == 'val':
            # Epoch 2 has higher accuracy but identical NLL; epoch 3 has worse NLL.
            losses = [2., .4, .4, .6, .4]
            value = losses[len(val_calls)]
            accuracy = [.1, .6, .9, 1., .6][len(val_calls)]
            val_calls.append(value)
        else:
            value, accuracy = .7, .5
        return {'balanced_nll': value, 'accuracy': accuracy}, [{'sample_id': 'fixture'}]

    monkeypatch.setattr(adapter.t.k, 'load_data', load_data)
    monkeypatch.setattr(adapter.t, 'build', lambda *args: TinyModel())
    monkeypatch.setattr(adapter.b, 'forward', forward)
    monkeypatch.setattr(adapter.b, 'encode', lambda x, *args: x)
    monkeypatch.setattr(adapter.b, 'affine_parameters', lambda *args: torch.zeros(7))
    monkeypatch.setattr(adapter.b, 'phase_smoothness', lambda model: model.raw_phase.square().mean())
    monkeypatch.setattr(adapter.b, 'evaluate', evaluate)
    monkeypatch.setattr(adapter.torch.cuda, 'empty_cache', lambda: None)
    adapter.train(a, profile, manifest, cfg, {'fixture_source': 'locked'})
    result = json.loads((a.out / 'result.json').read_text())
    history = json.loads((a.out / 'moe_L2_seed17/history.json').read_text())
    assert loaded == ['train', 'val', 'test'] and test_calls == ['once']
    assert all(not any('test' in key for key in row) for row in history)
    assert result['selected_epoch'] == 1 and result['selected_state_kind'] == 'ema'
    assert result['test_used_for_selection'] is False and result['test_evaluations'] == 1
