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


@pytest.mark.parametrize('profiles', [('routerfill_shared', 'qwen_shared'), ('routerfill_slim', 'qwen_slim'),
                                     ('routerfill_slim_norm', 'qwen_slim_norm')])
def test_exact_same_head_initialization_and_outputs(profiles):
    ours = load_settings(ROOT / f'configs/{profiles[0]}.yaml')
    baseline = load_settings(ROOT / f'configs/{profiles[1]}.yaml')
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


@pytest.mark.parametrize('profile', ['qwen_shared', 'qwen_slim', 'qwen_slim_norm'])
def test_baseline_forward_no_extra_mixer(profile):
    from LightGenV2.tasks.t04_semantic_interaction.training import _phase_regularization
    cfg = load_settings(ROOT / f'configs/{profile}.yaml')
    model = QwenSharedReadout(cfg)
    out = model(torch.randn(2, 196, 1024), [torch.randn(4, 2048), torch.randn(12, 2048)])
    assert out['category_logits'].shape == (2, 17, 6, 6)
    assert set(model._modules) == {'vision_stem', 'image_adapter', 'text_adapter', 'shared_readout'}
    (out['category_logits'].square().mean() + out['edit_logits'].square().mean()).backward()
    assert model.image_adapter[0].weight.grad is not None
    assert _phase_regularization(model, cfg).item() == 0.0
    # A nonzero optical regularizer on a non-optical model remains an error.
    with pytest.raises(RuntimeError, match='PhaseLayer'):
        _phase_regularization(model, SimpleNamespace(phase_dc_weight=1.0))


def test_slim_contract_and_parameter_budget():
    from LightGenV2.tasks.t04_semantic_interaction.shared_readout import SharedGridReadout
    standard, slim = SharedGridReadout(), SharedGridReadout(variant='slim')
    assert head_signature(standard)['parameters'] == 381976
    assert head_signature(slim)['parameters'] == 268888
    assert not hasattr(slim, 'post_film')
    assert isinstance(slim.decoder.pre, torch.nn.Identity)
    assert len(slim.editor) == 2
    norm = SharedGridReadout(variant='slim_norm')
    assert head_signature(norm)['parameters'] == 269272
    assert not any(isinstance(m, torch.nn.Conv2d) for m in norm.decoder.pre.modules())
    with pytest.raises(RuntimeError):
        slim.load_state_dict(standard.state_dict(), strict=True)


def test_instruction_span_is_exact_not_chat_summary():
    assert instruction_span([9, 8, 1, 2, 3, 7], [1, 2, 3]) == slice(2, 5)
    with pytest.raises(ValueError):
        instruction_span([1, 2, 1, 2], [1, 2])
