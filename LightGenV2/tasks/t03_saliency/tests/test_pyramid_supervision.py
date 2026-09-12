from pathlib import Path
import pytest
import torch
from LightGenV2.tasks.t03_saliency.pyramid_supervision import loss, validate


def test_identity_scale_invariance_and_target_detached():
    target = (torch.rand(2, 1, 8, 8, dtype=torch.float64) + .1).requires_grad_()
    logits = target.detach().log().requires_grad_()
    value, pieces = loss(logits, target, [8, 4, 2])
    assert abs(value.item()) < 1e-12
    scaled, _ = loss(logits + 20, target * 77, [8, 4, 2])
    torch.testing.assert_close(value, scaled, atol=1e-12, rtol=0)
    value.backward()
    assert target.grad is None and torch.isfinite(logits.grad).all()
    assert set(pieces) == {'pyramid_cc_loss_8', 'pyramid_cc_loss_4', 'pyramid_cc_loss_2'}


def test_constant_targets_skipped_and_constant_prediction_finite():
    logits = torch.zeros(2, 1, 8, 8, requires_grad=True)
    value, _ = loss(logits, torch.ones_like(logits), [4, 2])
    assert value.item() == 0
    value.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))
    logits.grad = None
    value, _ = loss(logits, torch.rand_like(logits) + .1, [4, 2])
    value.backward()
    assert torch.isfinite(logits.grad).all()


def test_gradcheck_and_pooling_probabilities():
    torch.manual_seed(19)
    logits = torch.randn(1, 1, 4, 4, dtype=torch.float64, requires_grad=True)
    target = torch.rand_like(logits) + .2
    assert torch.autograd.gradcheck(lambda x: loss(x, target, [4, 2])[0], logits)
    x = torch.nn.functional.avg_pool2d(logits.flatten(1).softmax(1).reshape_as(logits), 2).flatten()
    y = torch.nn.functional.avg_pool2d(target, 2).flatten()
    expected = 1 - torch.corrcoef(torch.stack([x, y]))[0, 1]
    torch.testing.assert_close(loss(logits, target, [2])[0], expected)


@pytest.mark.parametrize('options', [
    {'weight': float('nan'), 'sizes': [4]}, {'weight': 1, 'sizes': [3]},
    {'weight': 1, 'sizes': [4, 4]}, {'weight': 1, 'sizes': [True]},
    {'weight': 1, 'sizes': []}, {'weight': 1, 'sizes': [4], 'extra': 0},
])
def test_invalid_options(options):
    with pytest.raises(ValueError):
        validate(options, 8)


def test_matched_profiles_preserve_inference_and_serialization(tmp_path):
    import yaml
    from LightGenV2.tasks.t03_saliency.settings import load_settings, save_resolved_config
    from LightGenV2.tasks.t03_saliency.modeling import architecture_label
    root = Path(__file__).resolve().parents[1] / 'configs'
    original = load_settings(root / 'moe_alpha40_extra_control.yaml')
    a = load_settings(root / 'moe_alpha40_cc_fullgrid_20260913.yaml')
    b = load_settings(root / 'moe_alpha40_cc_pyramid_20260913.yaml')
    assert architecture_label(a) == architecture_label(b) == architecture_label(original)
    for key in ['initialization_checkpoint_sha256', 'student_learning_rate', 'phase_learning_rate',
                'router_learning_rate', 'ema_decay', 'fusion_alpha_min', 'router_backend', 'top_k',
                'language_optical_phase_zero_order_intensity_min', 'language_optical_phase_zero_order_intensity_max']:
        assert getattr(a, key) == getattr(b, key) == getattr(original, key)
    assert a.student_epochs == b.student_epochs == 30
    assert a.pyramid_cc == {'weight': 1.5, 'sizes': [224]}
    assert b.pyramid_cc == {'weight': 1.5, 'sizes': [56, 14]}
    b.output_dir = tmp_path
    save_resolved_config(b)
    assert yaml.safe_load((tmp_path/'resolved_config.yaml').read_text())['loss']['pyramid_cc'] == b.pyramid_cc


def test_training_hook_adds_only_requested_term():
    from types import SimpleNamespace
    from LightGenV2.tasks.t03_saliency.training_support import task_saliency_loss
    settings = SimpleNamespace(distillation_loss='spatial_cc', map_kd_temperature=1,
        kl_weight=1., cc_weight=1.5, sim_weight=.25, nss_weight=.1, map_kd_weight=2.)
    logits = torch.randn(2, 1, 8, 8, requires_grad=True)
    target, fixation = torch.rand_like(logits) + .1, torch.ones_like(logits)
    teacher = torch.randn_like(logits)
    base, pieces = task_saliency_loss(logits, target, fixation, settings, teacher_logits=teacher)
    settings.pyramid_cc = {'weight': 1.5, 'sizes': [4, 2]}
    total, updated = task_saliency_loss(logits, target, fixation, settings, teacher_logits=teacher)
    torch.testing.assert_close(total, base + 1.5 * loss(logits, target, [4, 2])[0])
    for key in pieces:
        torch.testing.assert_close(pieces[key], updated[key])
    total.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
