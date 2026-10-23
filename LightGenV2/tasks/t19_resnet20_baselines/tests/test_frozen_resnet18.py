"""Offline tests of pretrained selection, frozen tensors, BN and head updates."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torchvision import models as tv_models

from LightGenV2.tasks.t19_resnet20_baselines import frozen_resnet18 as frozen


@pytest.fixture
def offline_encoder(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    original_constructor = tv_models.resnet18
    reference = original_constructor(weights=None)
    reference.fc = nn.Identity()
    expected_state = frozen.encoder_state_sha256(reference)
    calls = []

    def official_constructor(*, weights):
        calls.append(weights)
        return reference

    monkeypatch.setattr(tv_models, "resnet18", official_constructor)
    monkeypatch.setattr(frozen, "_checkpoint_identity", lambda model, weights: {
        "weights_enum": "ResNet18_Weights.IMAGENET1K_V1",
        "weights_url": weights.url,
        "checkpoint_sha256": "offline_fixture_no_official_checkpoint",
    })
    encoder = frozen.FrozenResNet18()
    yield encoder, expected_state, calls
    torch.set_num_threads(previous)


def test_official_enum_and_compatible_state_keys(offline_encoder):
    encoder, expected_state, calls = offline_encoder
    assert calls == [tv_models.ResNet18_Weights.IMAGENET1K_V1]
    assert encoder.pretrained_provenance["encoder_parameters"] == 11176512
    assert encoder.pretrained_provenance["encoder_trainable_parameters"] == 0
    assert encoder.pretrained_provenance["encoder_state_sha256"] == expected_state
    assert "conv1.weight" in encoder.state_dict()
    assert "layer4.1.bn2.num_batches_tracked" in encoder.state_dict()
    assert not any(key.startswith("backbone.") for key in encoder.state_dict())


def test_task_train_updates_only_head_and_preserves_all_encoder_buffers(offline_encoder):
    encoder, expected_state, _ = offline_encoder
    model = nn.Sequential(encoder, nn.Linear(512, 5))
    head_before = model[1].weight.detach().clone()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.01)
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert optimizer_ids == {id(p) for p in model[1].parameters()}
    for _ in range(2):
        model.train()
        assert model[1].training
        assert all(not module.training for module in encoder.modules())
        images = torch.randn(2, 3, 32, 32, requires_grad=True)
        optimizer.zero_grad(set_to_none=True)
        model(images).square().mean().backward()
        optimizer.step()
        assert images.grad is None
        assert all(value.grad is None for value in encoder.parameters())
        audit = frozen.assert_encoder_unchanged(encoder, expected_state)
        assert audit["encoder_unchanged"]
    assert not torch.equal(model[1].weight, head_before)
    assert all(not value.requires_grad for value in encoder.parameters())
    features = encoder.features(torch.randn(1, 3, 64, 64, requires_grad=True))
    assert [value.shape[1] for value in features] == [64, 128, 256, 512]
    assert all(not value.requires_grad for value in features)


def test_audit_detects_buffer_parameter_gradient_and_mode_changes(offline_encoder):
    encoder, expected_state, _ = offline_encoder
    with torch.no_grad():
        encoder.bn1.num_batches_tracked.add_(1)
    with pytest.raises(RuntimeError, match="parameters or buffers changed"):
        frozen.assert_encoder_unchanged(encoder, expected_state)
    with torch.no_grad():
        encoder.bn1.num_batches_tracked.sub_(1)
    encoder.conv1.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="trainable parameters"):
        frozen.assert_encoder_unchanged(encoder, expected_state)
    encoder.conv1.weight.requires_grad_(False)
    encoder.conv1.weight.grad = torch.zeros_like(encoder.conv1.weight)
    with pytest.raises(RuntimeError, match="parameter gradients"):
        frozen.assert_encoder_unchanged(encoder, expected_state)
    encoder.conv1.weight.grad = None
    encoder.bn1.train()
    with pytest.raises(RuntimeError, match="training mode"):
        frozen.assert_encoder_unchanged(encoder, expected_state)
    encoder.train()
    assert frozen.assert_encoder_unchanged(encoder, expected_state)["encoder_all_modules_eval"]


def test_checkpoint_identity_hashes_real_file_and_checks_loaded_tensors(tmp_path, monkeypatch):
    encoder = nn.Linear(2, 3)
    cache = tmp_path / "checkpoints"
    cache.mkdir()
    temporary = cache / "fixture.pth"
    torch.save(encoder.state_dict(), temporary)
    digest = frozen.file_sha256(temporary)
    checkpoint = cache / f"resnet18-{digest[:8]}.pth"
    checkpoint.write_bytes(temporary.read_bytes())
    weights = SimpleNamespace(url=f"https://download.pytorch.org/models/{checkpoint.name}",
                              name="IMAGENET1K_V1")
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    monkeypatch.setattr(frozen, "OFFICIAL_CHECKPOINT_SHA256", digest)
    identity = frozen._checkpoint_identity(encoder, weights)
    assert identity["checkpoint_sha256"] == digest
    assert Path(identity["checkpoint_cache_path"]) == checkpoint.resolve()
    assert identity["checkpoint_tensors_match_encoder"]
    assert identity["checkpoint_full_sha256_verified"]
    assert identity["torchvision_restored_zero_bn_counters"] == []
    with torch.no_grad():
        encoder.weight.add_(1)
    with pytest.raises(RuntimeError, match="does not match pretrained checkpoint"):
        frozen._checkpoint_identity(encoder, weights)
    checkpoint.write_bytes(b"corrupted cache")
    with pytest.raises(RuntimeError, match="URL SHA256 prefix"):
        frozen._checkpoint_identity(encoder, weights)


def test_official_legacy_bn_counter_compatibility_is_narrow(tmp_path, monkeypatch):
    encoder = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4))
    cache = tmp_path / "checkpoints"
    cache.mkdir()
    state = encoder.state_dict()
    del state["1.num_batches_tracked"]
    temporary = cache / "legacy_fixture.pth"
    torch.save(state, temporary)
    digest = frozen.file_sha256(temporary)
    checkpoint = cache / f"resnet18-{digest[:8]}.pth"
    checkpoint.write_bytes(temporary.read_bytes())
    weights = SimpleNamespace(url=f"https://download.pytorch.org/models/{checkpoint.name}",
                              name="IMAGENET1K_V1")
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    monkeypatch.setattr(frozen, "OFFICIAL_CHECKPOINT_SHA256", digest)
    assert frozen._checkpoint_identity(encoder, weights)["torchvision_restored_zero_bn_counters"] == [
        "1.num_batches_tracked"]
    encoder[1].num_batches_tracked.add_(1)
    with pytest.raises(RuntimeError, match="1.num_batches_tracked"):
        frozen._checkpoint_identity(encoder, weights)
    encoder[1].num_batches_tracked.zero_()
    with torch.no_grad():
        encoder[1].running_mean.add_(1)
    with pytest.raises(RuntimeError, match="1.running_mean"):
        frozen._checkpoint_identity(encoder, weights)
    with torch.no_grad():
        encoder[1].running_mean.zero_()
        encoder[0].weight.add_(1)
    with pytest.raises(RuntimeError, match="0.weight"):
        frozen._checkpoint_identity(encoder, weights)
    monkeypatch.setattr(frozen, "OFFICIAL_CHECKPOINT_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="full SHA256"):
        frozen._checkpoint_identity(encoder, weights)


def test_state_hash_includes_names_dtype_shape_and_scalar_buffers():
    left, right = nn.BatchNorm2d(2), nn.BatchNorm2d(2)
    assert frozen.encoder_state_sha256(left) == frozen.encoder_state_sha256(right)
    right.num_batches_tracked.add_(1)
    assert frozen.encoder_state_sha256(left) != frozen.encoder_state_sha256(right)
    assert frozen.encoder_state_sha256(nn.Linear(2, 2)) != frozen.encoder_state_sha256(nn.Linear(1, 4))
