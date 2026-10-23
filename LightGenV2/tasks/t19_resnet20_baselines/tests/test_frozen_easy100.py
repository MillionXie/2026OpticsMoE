"""CPU-only training-contract checks; no official weights, data or CUDA."""
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from LightGenV2.tasks.t19_resnet20_baselines import train_easy100 as trainer


class TinyEncoder(nn.Module):
    def __init__(self, inputs, frozen):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(inputs, 3), nn.BatchNorm1d(3))
        self.frozen = frozen
        self.pretrained_provenance = {'offline_fixture': True, 'encoder_inputs': inputs}
        if frozen:
            self.requires_grad_(False)
            self.train(False)

    def train(self, mode=True):
        return super().train(False if self.frozen else mode)

    def forward(self, values):
        if self.frozen:
            with torch.no_grad():
                return self.network(values.float())
        return self.network(values.float())


class TinyRetrieval(nn.Module):
    def __init__(self, frozen=True):
        super().__init__()
        self.encoder = TinyEncoder(4, frozen)
        self.text_encoder = TinyEncoder(3, frozen)
        self.projection = nn.Linear(3, 2)
        self.text_projection = nn.Linear(3, 2)

    def forward(self, images):
        return F.normalize(self.projection(self.encoder(images)), dim=-1)

    def encode_text(self, tokens):
        return F.normalize(self.text_projection(self.text_encoder(tokens)), dim=-1)


def test_dual_encoder_audit_detects_text_buffer_gradient_and_mode_drift():
    model = TinyRetrieval()
    expected = trainer.frozen_encoder_hashes(model)
    assert set(expected) == {'visual', 'text'}
    audits = trainer.audit_frozen_encoders(model, expected)
    assert all(audit['encoder_unchanged'] for audit in audits.values())
    model.train()
    assert all(not encoder.training for encoder in (model.encoder, model.text_encoder))
    model.text_encoder.network[1].num_batches_tracked.add_(1)
    with pytest.raises(RuntimeError, match='parameters or buffers changed'):
        trainer.audit_frozen_encoders(model, expected)
    model.text_encoder.network[1].num_batches_tracked.sub_(1)
    model.text_encoder.network[0].weight.grad = torch.zeros_like(model.text_encoder.network[0].weight)
    with pytest.raises(RuntimeError, match='parameter gradients'):
        trainer.audit_frozen_encoders(model, expected)
    model.text_encoder.network[0].weight.grad = None
    model.text_encoder.network[1].train()
    with pytest.raises(RuntimeError, match='training mode'):
        trainer.audit_frozen_encoders(model, expected)


class FakeDataset:
    def __init__(self):
        self.train = [0, 1]
        self.test = [0, 1]
        self.titles = ['alpha', 'beta']
        self.title_skus = ['a', 'b']
        self.title_tokens = torch.tensor([[1, 0, 0], [0, 1, 0]])
        self.identity = {'offline_fixture': True}

    def images(self, split):
        return [{'image': torch.tensor([1., 0., 0., 1.]), 'sku': 'a', 'sample_id': split + '_a'},
                {'image': torch.tensor([0., 1., 1., 0.]), 'sku': 'b', 'sample_id': split + '_b'}]


class CpuTorch:
    """Redirect this runner's selected device without patching torch globally."""
    cuda = SimpleNamespace(get_device_name=lambda: 'CPU offline fixture', empty_cache=lambda: None)

    def __getattr__(self, name):
        return getattr(torch, name)

    @staticmethod
    def device(value):
        assert value == 'cuda:0'
        return torch.device('cpu')


@pytest.mark.parametrize('direction', ['image_to_text', 'text_to_image'])
@pytest.mark.parametrize('backbone', ['resnet18_imagenet_frozen', 'resnet20', 'resnet32'])
def test_full_runner_fixed_last_and_encoder_scope_without_cuda(tmp_path, monkeypatch, backbone, direction):
    frozen = backbone == 'resnet18_imagenet_frozen'
    model = TinyRetrieval(frozen=frozen)
    initial_hashes = trainer.frozen_encoder_hashes(model)
    head_before = model.projection.weight.detach().clone()
    config = tmp_path / 'configs'
    config.mkdir()
    (config / 'execution_authorization_20261023.json').write_text(json.dumps({
        'training_authorized_after_required_preflight': True}))
    run_dir = tmp_path / 'runs' / 'simulation' / (backbone + '_' + direction)
    run_dir.parent.mkdir(parents=True)
    monkeypatch.setattr(trainer, 'TASK', tmp_path)
    monkeypatch.setattr(trainer, 'require_published_source', lambda: None)
    monkeypatch.setattr(trainer, 'git', lambda *args: 'offline_main_commit')
    monkeypatch.setattr(trainer, 'Easy100', lambda root: FakeDataset())
    monkeypatch.setattr(trainer, 'wait_for_gpu', lambda args: [])
    monkeypatch.setattr(trainer, 'build_model', lambda task, backbone: model)
    monkeypatch.setattr(trainer, 'torch', CpuTorch())
    # Exercise training and evidence contents, not pre-existing atomic rename
    # helpers. Windows/OneDrive can briefly lock targets during rapid epochs.
    monkeypatch.setattr(trainer, 'write_json', lambda path, value: path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8'))
    monkeypatch.setattr(trainer, 'save_checkpoint', lambda path, value: torch.save(value, path))
    args = SimpleNamespace(direction=direction, backbone=backbone, data_root=tmp_path,
        run_dir=run_dir, gpu_uuid='GPU-offline', seed=42, epochs=30,
        learning_rate=.001, batch_size=32, workers=0, smoke_batches=0,
        queue_status_path=None, shared_owner_pids=[])
    previous_threads = torch.get_num_threads()
    try:
        trainer.run(args)
    finally:
        torch.set_num_threads(previous_threads)
    manifest = json.loads((run_dir / 'run_manifest.json').read_text())
    result = json.loads((run_dir / 'result.json').read_text())
    last = torch.load(run_dir / 'last_checkpoint.pt', map_location='cpu', weights_only=False)
    assert manifest['status'] == result['status'] == 'complete'
    assert result['selected_epoch'] == last['epoch'] == 30
    assert result['selection'] == 'fixed_last_epoch'
    assert result['selection_uses_test'] is False
    assert result['checkpoint_sha256'] == hashlib.sha256((run_dir / 'last_checkpoint.pt').read_bytes()).hexdigest()
    assert len(json.loads((run_dir / 'history.json').read_text())) == 30
    predictions = json.loads((run_dir / 'fixed_weight_predictions.json').read_text())
    assert len(predictions['query_ids']) == 2
    assert len(predictions['candidate_ids']) == 2
    assert not torch.equal(model.projection.weight, head_before)
    expected_scope = 'decoder_only' if frozen else 'full_network'
    assert manifest['training_scope'] == result['training_scope'] == expected_scope
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert manifest['optimizer_parameters'] == sum(p.numel() for p in trainable)
    assert len(last['optimizer']['param_groups'][0]['params']) == len(trainable)
    if frozen:
        assert manifest['encoder_state_sha256_before'] == manifest['encoder_state_sha256_after'] == initial_hashes
        assert len(manifest['frozen_encoder_audits']) == 30
        assert [audit['epoch'] for audit in manifest['frozen_encoder_audits']] == list(range(1, 31))
        assert set(manifest['pretrained_provenance']) == {'visual', 'text'}
        for audit in manifest['frozen_encoder_audits'] + [manifest['frozen_encoder_final_audit']]:
            assert all(audit[name]['encoder_unchanged'] for name in ('visual', 'text'))
        assert manifest['initialization'] == 'official_pretrained_visual_and_text'
    else:
        assert manifest['initialization'] == 'random_all_parameters_no_teacher'
        assert 'frozen_encoder_audits' not in manifest
        assert result['frozen_encoder_audit'] is None
        assert trainer.frozen_encoder_hashes(model) != initial_hashes


def test_frozen_profile_rejects_unapproved_lr_or_batch(monkeypatch):
    monkeypatch.setattr(trainer, 'require_published_source', lambda: None)
    args = SimpleNamespace(backbone='resnet18_imagenet_frozen', epochs=30, batch_size=16,
                           smoke_batches=0, learning_rate=.001, seed=42)
    with pytest.raises(ValueError, match='fixed lr0.001 and effective batch32'):
        trainer.run(args)
