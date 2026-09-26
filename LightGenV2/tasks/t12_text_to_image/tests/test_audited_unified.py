"""Physical geometry and language architecture regression guards."""
import torch
import pytest

from LightGenV2.tasks.t12_text_to_image.audited_unified import (
    audited_settings, optical_path, AuditedQwenTextEncoder, AuditedSpatialBottleneck,
)
from LightGenV2.tasks.t12_text_to_image.qwen_mini_small import QwenMiniConfig


def test_fixed_geometry_and_optical_router():
    settings = audited_settings()
    assert (settings.active_size, settings.canvas_size, settings.expert_size) == (478, 518, 224)
    path = optical_path(32, 64)
    assert path.core.router.active_phase().shape == (478, 478)
    assert path.core.router.raw_router_phase.shape == (224, 224)
    assert path.core.global_phase.phase.raw_phase.shape == (478, 478)
    assert len(path.core.expert_layers) == 1
    assert [tuple(e.raw_phase.shape) for e in path.core.expert_layers[0].experts] == [(224, 224)]*4
    phases = [p for name, p in path.named_parameters() if "raw_phase" in name or "raw_router_phase" in name]
    assert sum(p.numel() for p in phases) == 479364


def test_same_phases_for_different_electronic_widths():
    a, b = optical_path(32, 64), optical_path(64, 196)
    phase_shapes = lambda path: {name: tuple(p.shape) for name, p in path.named_parameters()
                               if "raw_phase" in name or "raw_router_phase" in name}
    assert phase_shapes(a) == phase_shapes(b)


def test_no_feature_grid_can_resize_physical_masks():
    with pytest.raises(ValueError):
        AuditedSpatialBottleneck(32, 16, grid=16)
    with pytest.raises(ValueError):
        optical_path(32, 225)


def test_language_keeps_two_qwen_style_blocks_and_alpha_floor():
    model = AuditedQwenTextEncoder(QwenMiniConfig(input_width=16, width=32,
                                               intermediate_width=64, heads=4, max_length=8))
    assert len(model.layers) == 2
    assert model.layers[0].gate_proj.weight.shape == (64, 32)
    for fusion in (model.fusion1, model.fusion2):
        with torch.no_grad():
            fusion.raw_alpha.fill_(-100)
        assert float(fusion.alpha) >= .4


def test_rank_reduction_preserves_matching_coefficient_rows():
    from types import SimpleNamespace
    from torch import nn
    from LightGenV2.tasks.t12_text_to_image.audited_unified import reduce_condition_rank
    adapter = nn.Module()
    adapter.predictor = nn.Sequential(nn.Linear(3, 5))
    adapter.register_buffer("basis", torch.randn(5, 8))
    adapter.register_buffer("coefficient_mean", torch.randn(5))
    adapter.register_buffer("coefficient_std", torch.ones(5))
    probe = torch.randn(2, 3)
    wanted = adapter.predictor(probe).detach()[:, :3]
    reduce_condition_rank(SimpleNamespace(adapter=adapter), 3)
    assert torch.allclose(adapter.predictor(probe), wanted)
    assert adapter.basis.shape == (3, 8)
    assert adapter.coefficient_mean.shape == (3,)


def test_detail_distillation_rejects_worse_teacher_and_backpropagates():
    from LightGenV2.tasks.t12_text_to_image.audited_unified_run import reliable_teacher_detail
    target = torch.zeros(2, 3, 16, 16)
    prediction = torch.full_like(target, .2).requires_grad_()
    assert float(reliable_teacher_detail(prediction, torch.ones_like(target), target)) == 0.
    loss = reliable_teacher_detail(prediction, target, target)
    assert float(loss) > 0.
    loss.backward()
    assert prediction.grad is not None and prediction.grad.abs().sum() > 0


def test_patchgan_generator_receives_gradient_with_frozen_discriminator():
    from LightGenV2.tasks.t12_text_to_image.audited_unified_run import DetailDiscriminator
    discriminator = DetailDiscriminator().eval().requires_grad_(False)
    reference = torch.randn(1, 3, 64, 64)
    image = torch.randn_like(reference, requires_grad=True)
    score = discriminator(reference, image)
    assert score.shape == (1, 1, 4, 4)
    (-score.mean()).backward()
    assert image.grad is not None and image.grad.abs().sum() > 0
    assert all(p.grad is None for p in discriminator.parameters())


def test_structured_text_pruning_preserves_recorded_neurons():
    from types import SimpleNamespace
    from LightGenV2.tasks.t12_text_to_image.audited_unified import prune_text_mlp
    from LightGenV2.tasks.t12_text_to_image.qwen_mini_small import QwenMiniTextEncoder
    config = QwenMiniConfig(input_width=16, width=32, intermediate_width=64, heads=4)
    text = QwenMiniTextEncoder(config)
    weights = text.layers[0].gate_proj.weight.detach().clone()
    model = SimpleNamespace(text=text)
    prune_text_mlp(model, 40)
    assert text.config.intermediate_width == 40
    assert torch.equal(text.layers[0].gate_proj.weight, weights[model.text_mlp_indices[0]])
    replica = SimpleNamespace(text=QwenMiniTextEncoder(config))
    prune_text_mlp(replica, 40, model.text_mlp_indices)
    replica.text.load_state_dict(text.state_dict(), strict=True)


def test_detail_crops_are_aligned_and_keep_student_gradients():
    from LightGenV2.tasks.t12_text_to_image.audited_unified_run import detail_patches
    reference = torch.zeros(2, 3, 256, 256)
    target = torch.randn_like(reference)
    prediction = target.clone().requires_grad_()
    gt, student = detail_patches(reference, target, prediction)
    assert gt.shape == (4, 3, 96, 96)
    assert torch.equal(gt, student)
    student.mean().backward()
    assert prediction.grad.abs().sum() > 0


def test_decoder_refinement_starts_as_identity_and_can_learn():
    from LightGenV2.tasks.t12_text_to_image.audited_unified import ConditionedDetailResidual
    block = ConditionedDetailResidual(32, 64, 16)
    value = torch.randn(2, 32, 8, 8)
    output = block(value, torch.randn(2, 16))
    assert torch.equal(output, value)
    output.square().mean().backward()
    assert block.conv2.weight.grad.abs().sum() > 0


def test_detail_expansion_parameter_budget_and_duplicate_guard():
    from types import SimpleNamespace
    from torch import nn
    from LightGenV2.tasks.t12_text_to_image.audited_unified import add_decoder_refinement
    editor = nn.Module()
    editor.config = SimpleNamespace(widths=(48,96,160,224), condition_dim=160)
    editor.up3, editor.up2 = nn.Identity(), nn.Identity()
    model = SimpleNamespace(kind="small", editor=editor)
    add_decoder_refinement(model)
    added = sum(p.numel() for p in editor.parameters())
    assert added == 4760352
    assert added + 9958098 < 15000000
    assert len(editor.up3.details) == 5 and len(editor.up2.details) == 2
    with pytest.raises(ValueError):
        add_decoder_refinement(model)
