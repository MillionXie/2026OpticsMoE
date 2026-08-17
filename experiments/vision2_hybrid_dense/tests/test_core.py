from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import torch
from torch import nn

from experiments.vision2_hybrid_dense.modeling import (
    LesionSegmentationDecoder,
    PoseHeatmapDecoder,
    SaliencyDensityDecoder,
    Vision2HybridDenseStudent,
    restore_qwen_block_major_spatial,
)
from experiments.vision2_hybrid_dense.settings import (
    apply_vision2_hybrid_settings,
)
from experiments.vision2_hybrid_dense.hardware_bridge import (
    _downstream_parameters,
    _optimizer,
)


def test_restore_qwen_block_major_spatial_exactly() -> None:
    raster = torch.arange(16, dtype=torch.float32).view(4, 4)
    packed = (
        raster.view(2, 2, 2, 2)
        .permute(0, 2, 1, 3)
        .reshape(16, 1)
    )
    restored = restore_qwen_block_major_spatial(
        packed, torch.tensor([[1, 4, 4]])
    )
    torch.testing.assert_close(restored[0, 0], raster)


def test_dense_decoders_have_expected_shapes_and_gradients() -> None:
    inputs = torch.randn(2, 192, 8, 8, requires_grad=True)
    saliency = SaliencyDensityDecoder(output_size=64)(inputs)
    lesion = LesionSegmentationDecoder(output_size=64)(inputs)
    pose = PoseHeatmapDecoder(heatmap_size=32)(inputs)
    assert saliency.shape == (2, 1, 64, 64)
    assert lesion.shape == (2, 1, 64, 64)
    assert pose.shape == (2, 14, 32, 32)
    (saliency.mean() + lesion.mean() + pose.mean()).backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


def test_hybrid_settings_use_trainable_but_conservative_phase_lr() -> None:
    settings = SimpleNamespace()
    apply_vision2_hybrid_settings(
        settings,
        {
            "vision2_hybrid": {
                "enabled": True,
                "optimization": {"phase_learning_rate": 1.0e-4},
            }
        },
    )
    assert settings.phase_learning_rate == 1.0e-4
    assert settings.num_experts == 4
    assert settings.top_k == 2
    assert settings.electronic_layers == 2
    assert settings.hardware_ccd_target_size == 478


def test_fake_qwen_stem_reaches_both_phase_masks() -> None:
    from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval.settings import (
        load_settings,
    )

    root = Path(__file__).resolve().parents[3]
    settings = load_settings(
        root
        / "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval"
        / "configs/release/caltech101_four_layer_optical_joint.yaml"
    )
    settings.vision_hidden_size = 10
    settings.electronic_width = 6
    settings.input_adapter_dim = 224
    settings.detector_output_size = 224
    settings.electronic_dropout = 0.0
    settings.phase_dropout_p = 0.0
    settings.language_optical_max_shift_pixels = 0
    settings.language_optical_phase_shift_pixels = 0
    settings.language_optical_ccd_shift_pixels = 0
    settings.k_space_constraint_enabled = False

    class Pass(nn.Module):
        def forward(self, value: torch.Tensor, **_: object) -> torch.Tensor:
            return value

    class Visual(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.patch_embed = nn.Linear(10, 10)
            self.blocks = nn.ModuleList([Pass(), Pass()])
            self.deepstack_visual_indexes = [0]

        def forward(
            self, values: torch.Tensor, *, grid_thw: torch.Tensor
        ) -> torch.Tensor:
            hidden = self.patch_embed(values)
            lengths = grid_thw.prod(1).long()
            cu = torch.cat((lengths.new_zeros(1), lengths.cumsum(0))).int()
            for block in self.blocks:
                hidden = block(hidden, cu_seqlens=cu)
            return hidden

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual = Visual()

    base = Model()
    loaded = SimpleNamespace(visual=base.visual, model=base, device=torch.device("cpu"))
    head = nn.Sequential(nn.Conv2d(6, 1, 1), nn.AdaptiveAvgPool2d(1))
    student = Vision2HybridDenseStudent(loaded, settings, head).train()
    output, spatial, detector = student(
        torch.randn(4, 10), torch.tensor([[1, 2, 2]])
    )
    assert output.shape == (1, 1, 1, 1)
    assert spatial.shape == (1, 6, 2, 2)
    assert detector.shape == (1, 224, 224)
    output.square().mean().backward()
    phase_parameters = [
        parameter
        for name, parameter in student.core.named_parameters()
        if "raw_phase" in name
    ]
    assert phase_parameters
    assert all(parameter.grad is not None for parameter in phase_parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in phase_parameters)
    assert student.core.hybrid.block1_optical_fusion_logit.grad is not None
    assert student.core.hybrid.block2_optical_fusion_logit.grad is not None
    settings.hardware_finetune_learning_rate = 5.0e-5
    settings.dense_readout_learning_rate = 5.0e-5
    settings.dense_head_learning_rate = 3.0e-4
    settings.phase_learning_rate = 1.0e-4
    context = SimpleNamespace(model=student, settings=settings)
    optimizer = _optimizer(
        context, _downstream_parameters(context, "vision_expert")
    )
    learning_rates = {
        group["name"]: group["lr"] for group in optimizer.param_groups
    }
    assert learning_rates["phase"] == 1.0e-4
    assert learning_rates["readout"] == 5.0e-5
    assert learning_rates["head"] == 3.0e-4
    student.restore_native()
