from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from LightGenV2.tasks.t04_semantic_interaction.router_repair import full_aperture_language_amplitude
from LightGenV2.tasks.t04_semantic_interaction.shared_readout import create_shared_readout, head_signature
from LightGenV2.tasks.t04_semantic_interaction.qwen_shared import QwenSharedReadout, instruction_span
from LightGenV2.tasks.t04_semantic_interaction.settings import load_settings

ROOT = Path(__file__).resolve().parents[1]


def test_router_fill_keeps_power_and_has_gradients():
    fields = torch.zeros(2, 16, 16)
    fields[0, :3] = torch.linspace(.2, 1., 16)
    fields[1, :7] = .3
    fields.requires_grad_()
    filled = full_aperture_language_amplitude(fields)
    assert filled.shape == fields.shape and (filled[:, -1].sum(-1) > 0).all()
    torch.testing.assert_close(filled.square().sum((-2, -1)), fields.square().sum((-2, -1)))
    torch.testing.assert_close(full_aperture_language_amplitude(fields.flip(0)).flip(0), filled)
    filled.sum().backward()
    assert torch.isfinite(fields.grad).all()


def test_exact_same_head_initialization_and_outputs():
    ours = load_settings(ROOT / 'configs/routerfill_shared.yaml')
    baseline = load_settings(ROOT / 'configs/qwen_shared.yaml')
    a, b = create_shared_readout(ours), create_shared_readout(baseline)
    assert head_signature(a) == head_signature(b)
    assert len(a.editor) == len(b.editor) == 2
    x, condition = torch.randn(2, 192, 14, 14), torch.randn(2, 192)
    ya, yb = a(x, condition), b(x, condition)
    for name in ya:
        torch.testing.assert_close(ya[name], yb[name], rtol=0, atol=0)
    assert ours.changed_cell_weight == baseline.changed_cell_weight
    assert ours.category_loss_weight == baseline.category_loss_weight
    assert ours.task_loss_weight == baseline.task_loss_weight
    assert ours.epochs == baseline.epochs == 100


def test_baseline_forward_no_extra_mixer():
    cfg = load_settings(ROOT / 'configs/qwen_shared.yaml')
    model = QwenSharedReadout(cfg)
    out = model(torch.randn(2, 196, 1024), [torch.randn(4, 2048), torch.randn(12, 2048)])
    assert out['category_logits'].shape == (2, 17, 6, 6)
    assert set(model._modules) == {'vision_stem', 'image_adapter', 'text_adapter', 'shared_readout'}
    (out['category_logits'].square().mean() + out['edit_logits'].square().mean()).backward()
    assert model.image_adapter[0].weight.grad is not None


def test_instruction_span_is_exact_not_chat_summary():
    assert instruction_span([9, 8, 1, 2, 3, 7], [1, 2, 3]) == slice(2, 5)
    with pytest.raises(ValueError):
        instruction_span([1, 2, 1, 2], [1, 2])
