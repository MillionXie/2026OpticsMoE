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


def test_frozen_queue_dispatches_text_tasks_with_explicit_frozen_profile(tmp_path):
    for task in ['abo_image_to_text', 'abo_text_to_image', 'openmoji', 'abo_conditional_edit']:
        command = worker_command({'task': task, 'phase': 'formal', 'name': 'example'},
                                 tmp_path, 'GPU-example', 'resnet18_imagenet_frozen')
        assert command[command.index('--backbone')+1] == 'resnet18_imagenet_frozen'
        assert command[2].endswith('train_easy100' if task.startswith('abo_') and task.endswith(('to_text','text_to_image')) else 'train_tasks')


@pytest.mark.parametrize('task', ['abo_image_to_text', 'abo_text_to_image', 'openmoji', 'abo_conditional_edit'])
def test_multimodal_decoders_update_with_both_encoders_frozen(offline_pretrained, monkeypatch, task):
    from LightGenV2.tasks.t19_resnet20_baselines import frozen_multimodal_models as multimodal
    from LightGenV2.tasks.t19_resnet20_baselines.models import tokenize_text
    class OfflineText(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(259, 384)
            self.requires_grad_(False)
            self.train(False)
        def train(self, mode=True):
            super().train(False)
            return self
        @torch.no_grad()
        def forward(self, tokens):
            return self.embedding(tokens).mean(1)
    monkeypatch.setattr(multimodal, 'FrozenTextEncoder', OfflineText)
    model = build_model(task, backbone='resnet18_imagenet_frozen')
    digests = [frozen.encoder_state_sha256(m) for m in (model.encoder, model.text_encoder)]
    trainable = [p for p in model.parameters() if p.requires_grad]
    before = [p.detach().clone() for p in trainable]
    optimizer = torch.optim.AdamW(trainable, lr=.001)
    model.train()
    images = torch.randn(2, 3, 32, 32)
    tokens = tokenize_text(['Move the red object left.', 'Change the object to blue.'])
    if task in {'abo_image_to_text', 'abo_text_to_image'}:
        output = model(images)
        text = model.encode_text(tokens)
        loss = (output - text).square().mean()
        assert output.shape == text.shape == (2,64)
    elif task == 'openmoji':
        output = model(images,tokens)
        assert output['category_logits'].shape == (2,17,6,6)
        assert output['edit_logits'].shape == (2,6,6)
        assert output['task_logits'].shape == (2,4)
        loss = sum((value-torch.randn_like(value)).square().mean() for value in output.values())
    else:
        output = model(images,tokens)
        assert output.shape == images.shape and output.min() >= 0 and output.max() <= 1
        loss = (output-torch.rand_like(output)).square().mean()
    loss.backward(); optimizer.step()
    for encoder,digest in zip((model.encoder, model.text_encoder),digests):
        assert frozen.assert_encoder_unchanged(encoder,digest)['encoder_unchanged']
    assert any(not torch.equal(p,initial) for p,initial in zip(trainable,before))
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
