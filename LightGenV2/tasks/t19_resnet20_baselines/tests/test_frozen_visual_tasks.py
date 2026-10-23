import pytest
import torch
from torch import nn
from torchvision import models as tv

from LightGenV2.tasks.t19_resnet20_baselines import frozen_resnet18 as frozen
from LightGenV2.tasks.t19_resnet20_baselines.models import build_model, parameter_counts
from LightGenV2.tasks.t19_resnet20_baselines.queue_remaining import worker_command


@pytest.fixture
def offline_pretrained(monkeypatch):
    constructor = tv.resnet18
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    def build(*, weights):
        assert weights is tv.ResNet18_Weights.IMAGENET1K_V1
        return constructor(weights=None)
    monkeypatch.setattr(tv, 'resnet18', build)
    monkeypatch.setattr(frozen, '_checkpoint_identity', lambda *args: {'offline_fixture': True})
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('task,head_count', [
    ('lgvq_temporal', 229765), ('lgvq_spatial', 33157),
    ('abo_image_to_image', 32832), ('lsp', 40558), ('salicon', 40129)])
def test_five_decoders_train_without_changing_any_encoder_state(offline_pretrained, task, head_count):
    model = build_model(task, backbone='resnet18_imagenet_frozen')
    counts = parameter_counts(model)
    assert counts['visual_encoder'] == 11176512
    assert counts['trainable'] == counts['task_heads'] == head_count
    digest = frozen.encoder_state_sha256(model.encoder)
    head_before = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=.001)
    model.train()
    images = torch.randn(2, 3, 32, 32)
    if task.startswith('lgvq_'):
        output = model(images[:, None].repeat(1, 4, 1, 1, 1))['quality_logits']
        assert output.shape == (2, 5)
    else:
        output = model(images)
        expected = {'abo_image_to_image': (2, 64), 'lsp': (2, 14, 56, 56), 'salicon': (2, 1, 224, 224)}
        assert output.shape == expected[task]
    # Random targets exercise gradients also through the L2 retrieval projection.
    loss = (output - torch.randn_like(output)).square().mean()
    loss.backward()
    optimizer.step()
    assert frozen.assert_encoder_unchanged(model.encoder, digest)['encoder_unchanged']
    assert any(not torch.equal(p, head_before[name])
               for name, p in model.named_parameters() if p.requires_grad)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)


def test_frozen_queue_cannot_silently_train_random_text_encoder(tmp_path):
    for task in ['abo_image_to_text', 'abo_text_to_image', 'openmoji', 'abo_conditional_edit']:
        with pytest.raises(ValueError, match='Text encoder protocol'):
            worker_command({'task': task, 'phase': 'formal', 'name': 'example'},
                           tmp_path, 'GPU-example', 'resnet18_imagenet_frozen')
        with pytest.raises(ValueError, match='text encoder protocol'):
            build_model(task, backbone='resnet18_imagenet_frozen')
