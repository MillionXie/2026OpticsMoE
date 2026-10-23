"""Small random byte-GRU + frozen visual encoder; offline CPU contracts."""
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from LightGenV2.tasks.t19_resnet20_baselines import train_easy100 as trainer
from LightGenV2.tasks.t19_resnet20_baselines.frozen_resnet18 import encoder_state_sha256
from LightGenV2.tasks.t19_resnet20_baselines.models import ByteTextEncoder, tokenize_text
from LightGenV2.tasks.t19_resnet20_baselines.tests.test_frozen_easy100 import (
    CpuTorch, FakeDataset, TinyEncoder)


class SmallTextRetrieval(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TinyEncoder(4, frozen=True)
        self.text_encoder = ByteTextEncoder()
        self.projection = nn.Linear(3, 8)
        self.text_projection = nn.Linear(64, 8)

    def forward(self, images):
        return F.normalize(self.projection(self.encoder(images)), dim=-1)

    def encode_text(self, tokens):
        return F.normalize(self.text_projection(self.text_encoder(tokens)), dim=-1)


@pytest.fixture
def small_text_runner(tmp_path, monkeypatch):
    torch.manual_seed(42)
    model = SmallTextRetrieval()
    dataset = FakeDataset()
    dataset.title_tokens = tokenize_text(['alpha product', 'beta product'])
    config = tmp_path / 'configs'
    config.mkdir()
    (config / 'execution_authorization_20261023.json').write_text(json.dumps({
        'training_authorized_after_required_preflight': True}))
    run_dir = tmp_path / 'runs' / 'simulation' / 'small_text'
    run_dir.parent.mkdir(parents=True)
    calls = []

    def build(task, **options):
        calls.append((task, options))
        return model

    monkeypatch.setattr(trainer, 'TASK', tmp_path)
    monkeypatch.setattr(trainer, 'require_published_source', lambda: None)
    monkeypatch.setattr(trainer, 'git', lambda *args: 'offline_main_commit')
    monkeypatch.setattr(trainer, 'Easy100', lambda root: dataset)
    monkeypatch.setattr(trainer, 'wait_for_gpu', lambda args: [])
    monkeypatch.setattr(trainer, 'build_model', build)
    monkeypatch.setattr(trainer, 'torch', CpuTorch())
    # The trainer's existing atomic filesystem helpers are outside this
    # protocol test; avoid transient Windows/OneDrive rename locks.
    monkeypatch.setattr(trainer, 'write_json', lambda path, value: path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8'))
    monkeypatch.setattr(trainer, 'save_checkpoint', lambda path, value: torch.save(value, path))
    args = SimpleNamespace(direction='image_to_text', backbone='resnet18_imagenet_frozen',
        text_encoder='byte_gru', data_root=tmp_path, run_dir=run_dir, gpu_uuid='GPU-offline',
        seed=42, epochs=30, learning_rate=.001, batch_size=32, workers=0, smoke_batches=0,
        queue_status_path=None, shared_owner_pids=[])
    previous_threads = torch.get_num_threads()
    yield SimpleNamespace(model=model, args=args, run_dir=run_dir, calls=calls)
    torch.set_num_threads(previous_threads)


@pytest.mark.parametrize('direction', ['image_to_text', 'text_to_image'])
def test_complete_30_epoch_run_freezes_visual_and_trains_all_small_text(small_text_runner, direction):
    fixture = small_text_runner
    fixture.args.direction = direction
    model = fixture.model
    visual_before = encoder_state_sha256(model.encoder)
    text_before = encoder_state_sha256(model.text_encoder)
    text_parameters_before = {name: value.detach().clone()
                              for name, value in model.text_encoder.named_parameters()}
    trainer.run(fixture.args)
    manifest = json.loads((fixture.run_dir / 'run_manifest.json').read_text())
    result = json.loads((fixture.run_dir / 'result.json').read_text())
    checkpoint = torch.load(fixture.run_dir / 'last_checkpoint.pt', map_location='cpu', weights_only=False)
    assert fixture.calls == [('abo_' + direction, {
        'backbone': 'resnet18_imagenet_frozen', 'text_encoder': 'byte_gru'})]
    assert manifest['status'] == result['status'] == 'complete'
    assert checkpoint['epoch'] == result['selected_epoch'] == 30
    assert result['selection'] == 'fixed_last_epoch' and result['selection_uses_test'] is False
    assert result['checkpoint_sha256'] == hashlib.sha256((fixture.run_dir / 'last_checkpoint.pt').read_bytes()).hexdigest()
    assert manifest['initialization'] == 'official_pretrained_visual_random_byte_gru'
    assert manifest['training_scope'] == result['training_scope'] == 'frozen_visual_trainable_text_and_decoder'
    assert manifest['text_encoder'] == 'byte_gru'
    assert set(manifest['pretrained_provenance']) == {'visual'}
    assert manifest['text_encoder_initialization']['initialization'] == 'random_no_pretraining'
    assert manifest['parameters']['text_encoder'] == 27_104
    assert manifest['parameters']['total'] < 15_000_000
    assert manifest['encoder_state_sha256_before'] == manifest['encoder_state_sha256_after'] == {'visual': visual_before}
    assert encoder_state_sha256(model.encoder) == visual_before
    assert manifest['trainable_text_state_sha256_before'] == text_before
    assert manifest['trainable_text_state_sha256_after'] == encoder_state_sha256(model.text_encoder) != text_before
    assert all(value.requires_grad for value in model.text_encoder.parameters())
    assert all(not torch.equal(value, text_parameters_before[name])
               for name, value in model.text_encoder.named_parameters())
    assert all(value.grad is None for value in model.encoder.parameters())
    assert len(manifest['frozen_encoder_audits']) == 30
    for audit in manifest['frozen_encoder_audits'] + [manifest['frozen_encoder_final_audit']]:
        assert 'text' not in audit
        assert audit['visual']['encoder_unchanged']
    trainable = [value for value in model.parameters() if value.requires_grad]
    assert manifest['optimizer_parameters'] == sum(value.numel() for value in trainable)
    assert manifest['optimizer_parameters'] == manifest['parameters']['text_encoder'] + manifest['parameters']['task_heads']
    assert len(checkpoint['optimizer']['param_groups'][0]['params']) == len(trainable)


def test_small_text_complete_parameter_cap_is_strict(small_text_runner, monkeypatch):
    fixture = small_text_runner
    original = trainer.parameter_counts
    monkeypatch.setattr(trainer, 'parameter_counts', lambda model: {
        **original(model), 'total': 15_000_000})
    with pytest.raises(ValueError, match='strictly below 15M'):
        trainer.run(fixture.args)
    assert not (fixture.run_dir / 'last_checkpoint.pt').exists()


def test_small_text_refuses_frozen_text_encoder(small_text_runner):
    fixture = small_text_runner
    fixture.model.text_encoder.requires_grad_(False)
    with pytest.raises(RuntimeError, match='must remain fully trainable'):
        trainer.run(fixture.args)
