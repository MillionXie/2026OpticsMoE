"""Offline checks for frozen vision plus a small trainable byte-GRU branch."""
import pytest
import torch
from torch import nn
from torchvision import models as tv_models

from LightGenV2.tasks.t19_resnet20_baselines import frozen_multimodal_models as multimodal
from LightGenV2.tasks.t19_resnet20_baselines import frozen_resnet18
from LightGenV2.tasks.t19_resnet20_baselines.frozen_text import decode_byte_tokens
from LightGenV2.tasks.t19_resnet20_baselines.models import (
    ByteTextEncoder, parameter_counts, tokenize_text,
)


@pytest.fixture
def offline_visual(monkeypatch):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    original_constructor = tv_models.resnet18

    def fake_official(*, weights):
        assert weights == tv_models.ResNet18_Weights.IMAGENET1K_V1
        return original_constructor(weights=None)

    monkeypatch.setattr(tv_models, 'resnet18', fake_official)
    monkeypatch.setattr(frozen_resnet18, '_checkpoint_identity', lambda model, weights: {
        'weights_enum': 'ResNet18_Weights.IMAGENET1K_V1',
        'weights_url': weights.url,
        'checkpoint_sha256': 'offline_fixture',
    })

    def refuse_text_download():
        raise AssertionError('byte_gru profile must not instantiate MiniLM')

    monkeypatch.setattr(multimodal, 'FrozenTextEncoder', refuse_text_download)
    yield
    torch.set_num_threads(previous_threads)


EXPECTED_COUNTS = {
    'abo_image_to_text': dict(total=11240608, trainable=64096, visual_encoder=11176512,
                              text_encoder=27104, task_heads=36992),
    'abo_text_to_image': dict(total=11240608, trainable=64096, visual_encoder=11176512,
                              text_encoder=27104, task_heads=36992),
    'openmoji': dict(total=11287478, trainable=110966, visual_encoder=11176512,
                     text_encoder=27104, task_heads=83862),
    'abo_conditional_edit': dict(total=11628515, trainable=452003, visual_encoder=11176512,
                                 text_encoder=27104, task_heads=424899),
}


@pytest.mark.parametrize('task', EXPECTED_COUNTS)
def test_small_text_four_tasks_train_only_text_and_decoder(offline_visual, task):
    torch.manual_seed(42)
    model = multimodal.build_frozen_multimodal_model(task, text_encoder='byte_gru')
    assert isinstance(model.text_encoder, ByteTextEncoder)
    counts = parameter_counts(model)
    assert counts == EXPECTED_COUNTS[task]
    assert counts['total'] < 15_000_000
    assert model.text_projection.in_features == 64
    initial_encoder = frozen_resnet18.encoder_state_sha256(model.encoder)
    text_before = model.text_encoder.gru.weight_ih_l0.detach().clone()
    projection_before = model.text_projection.weight.detach().clone()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.01)
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group['params']}
    assert optimizer_ids.isdisjoint({id(p) for p in model.encoder.parameters()})
    values = ['Move the BLUE cup 保留完整全文。', 'Add a red object.']
    tokens = tokenize_text(values)
    assert decode_byte_tokens(tokens) == values
    seen = []
    handle = model.text_encoder.register_forward_pre_hook(
        lambda module, args: seen.append(args[0].detach().clone()))
    model.train()
    images = torch.randn(2, 3, 32, 32)
    if task.startswith('abo_') and task.endswith(('image_to_text', 'text_to_image')):
        image_features, text_features = model(images), model.encode_text(tokens)
        assert image_features.shape == text_features.shape == (2, 64)
        loss = (image_features - text_features).square().mean()
    elif task == 'openmoji':
        output = model(images, tokens)
        assert output['category_logits'].shape == (2, 17, 6, 6)
        assert output['edit_logits'].shape == (2, 6, 6)
        assert output['task_logits'].shape == (2, 4)
        loss = sum(value.square().mean() for value in output.values())
    else:
        output = model(images, tokens)
        assert output.shape == images.shape
        assert bool(((output >= 0) & (output <= 1)).all())
        loss = (output - torch.rand_like(output)).square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert model.text_encoder.gru.weight_ih_l0.grad is not None
    assert model.text_encoder.gru.weight_ih_l0.grad.abs().sum().item() > 0
    assert model.text_projection.weight.grad is not None
    assert model.text_projection.weight.grad.abs().sum().item() > 0
    optimizer.step()
    handle.remove()
    assert len(seen) == 1 and torch.equal(seen[0], tokens)
    assert not torch.equal(model.text_encoder.gru.weight_ih_l0, text_before)
    assert not torch.equal(model.text_projection.weight, projection_before)
    assert frozen_resnet18.assert_encoder_unchanged(model.encoder, initial_encoder)['encoder_unchanged']


def test_original_default_still_selects_frozen_minilm(offline_visual, monkeypatch):
    class TinyFrozenText(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1), requires_grad=False)

    monkeypatch.setattr(multimodal, 'FrozenTextEncoder', TinyFrozenText)
    for cls in (multimodal.FrozenCrossModalRetrieval, multimodal.FrozenOpenMoji,
                multimodal.FrozenConditionalEditor):
        model = cls()
        assert isinstance(model.text_encoder, TinyFrozenText)
        assert model.text_projection.in_features == 384
    with pytest.raises(ValueError, match='text encoder profile'):
        multimodal._build_text_encoder('unknown')
