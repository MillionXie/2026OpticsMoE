from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from experiments.qwen3_vl_2b_cifar10_vision_homogeneous_moe9.datasets import RGBDataset
from experiments.qwen3_vl_2b_cifar10_vision_homogeneous_moe9.modeling import NormalizedLinearHead
from experiments.qwen3_vl_2b_cifar10_vision_homogeneous_moe9.optics.geometry import MoEGeometry
from experiments.qwen3_vl_2b_cifar10_vision_homogeneous_moe9.optics.moe import FullPlaneDetectorReadout, VisionHomogeneousMoESurrogate
from experiments.qwen3_vl_2b_cifar10_vision_homogeneous_moe9.optics.physical import PhaseLayer, SquareDetectionLayerNormReload
from experiments.qwen3_vl_2b_cifar10_vision_homogeneous_moe9.sampling import EpochClassMixedSampler
from experiments.qwen3_vl_2b_cifar10_vision_homogeneous_moe9.settings import load_settings
from experiments.qwen3_vl_2b_cifar10_vision_homogeneous_moe9.visualization import save_debug_example


CONFIG = "experiments/qwen3_vl_2b_cifar10_vision_homogeneous_moe9/configs/cifar10_smoke.json"


def _encoder(hidden_size: int = 8) -> VisionHomogeneousMoESurrogate:
    module = VisionHomogeneousMoESurrogate.__new__(VisionHomogeneousMoESurrogate)
    nn.Module.__init__(module)
    module.max_visual_tokens = 120
    module.geometry = MoEGeometry()
    module.input_adapter = nn.Linear(hidden_size, 120)
    module.input_norm = nn.LayerNorm(120)
    module.nonnegative = nn.Softplus()
    return module


def test_config_and_small_head() -> None:
    settings = load_settings(CONFIG)
    assert settings.detector_layernorm_affine is False
    assert settings.router_balance_weight == pytest.approx(0.1)
    assert settings.log_interval_batches == 1
    assert settings.optimizer_type == "adamw"
    assert settings.interlayer_hard_route_mask is True
    assert settings.interlayer_reapply_routing_weights is True
    assert settings.to_dict()["training"]["logging"]["interval_batches"] == 1
    head = NormalizedLinearHead(1024, 10)
    assert head(torch.randn(3, 1024)).shape == (3, 10)
    assert sum(parameter.numel() for parameter in head.parameters()) == 12298


def test_legacy_flat_config_remains_supported(tmp_path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"dataset": "cifar10", "log_interval_batches": 7}), encoding="utf-8")
    settings = load_settings(path)
    assert settings.log_interval_batches == 7
    assert settings.to_dict()["training"]["logging"]["interval_batches"] == 7


def test_epoch_sampler_rotates_balanced_class_windows() -> None:
    labels = [0] * 4 + [1] * 4 + [2] * 4
    sampler = EpochClassMixedSampler(range(12), labels, 3, batch_size=3, seed=5, per_class_limit=2)
    first = list(iter(sampler))
    sampler.set_epoch(2)
    second = list(iter(sampler))
    assert len(first) == len(second) == 6
    assert {class_index: sum(labels[index] == class_index for index in first) for class_index in range(3)} == {0: 2, 1: 2, 2: 2}
    assert set(first) != set(second)


def test_token_rows_zero_pad_without_resizing() -> None:
    encoder = _encoder()
    group = torch.randn(60, 8)
    field = encoder.encode_groups([group])
    assert field.shape == (1, 120, 120)
    assert torch.all(field >= 0)
    assert torch.count_nonzero(field[:, 60:]) == 0


def test_token_overflow_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="visual token count 121"):
        _encoder().encode_groups([torch.randn(121, 8)])


def test_full_detector_readout_is_non_affine_and_nonnegative() -> None:
    settings = SimpleNamespace(canvas_size=480, detector_pool_kernel=4, detector_layernorm_eps=1e-5,
                               detector_nonlinearity="relu")
    readout = FullPlaneDetectorReadout(settings)
    assert len(list(readout.norm.parameters())) == 0
    output, intensity = readout(torch.randn(2, 480, 480, dtype=torch.complex64))
    assert output.shape == (2, 120, 120)
    assert intensity.shape == (2, 480, 480)
    assert torch.all(output >= 0) and torch.all(intensity >= 0)


def test_interlayer_conversion_is_non_affine_and_keeps_shape() -> None:
    geometry = MoEGeometry()
    conversion = SquareDetectionLayerNormReload(480, geometry.expert_apertures, 1e-5, "relu", True, False)
    assert sum(parameter.numel() for parameter in conversion.parameters()) == 0
    field = torch.randn(1, 480, 480, dtype=torch.complex64)
    output = conversion(field)
    assert output.shape == field.shape and output.dtype == torch.complex64
    assert torch.all(output.real >= 0) and torch.count_nonzero(output.imag) == 0


def test_hard_route_mask_keeps_unselected_expert_fields_exactly_zero() -> None:
    geometry = MoEGeometry()
    conversion = SquareDetectionLayerNormReload(480, geometry.expert_apertures, 1e-5, "relu", True, False)
    selected = torch.zeros(2, 9, dtype=torch.bool)
    selected[0, [0, 4, 8]] = True
    selected[1, [1, 3, 5]] = True
    field = torch.randn(2, 480, 480, dtype=torch.complex64)
    output = conversion(field, selected)
    for sample in range(2):
        for expert, aperture in enumerate(geometry.expert_apertures):
            crop = output[sample, aperture.y0:aperture.y1, aperture.x0:aperture.x1]
            if selected[sample, expert]:
                assert torch.count_nonzero(crop.real) > 0
            else:
                assert torch.count_nonzero(crop) == 0


def test_routing_weights_are_restored_after_per_expert_normalization() -> None:
    geometry = MoEGeometry()
    conversion = SquareDetectionLayerNormReload(480, geometry.expert_apertures, 1e-5, "relu", True, False)
    field = torch.zeros(1, 480, 480, dtype=torch.complex64)
    source = torch.linspace(0.0, 1.0, 120 * 120).reshape(120, 120).to(torch.complex64)
    for expert in (0, 1):
        aperture = geometry.expert_apertures[expert]
        field[:, aperture.y0:aperture.y1, aperture.x0:aperture.x1] = source
    selected = torch.zeros(1, 9, dtype=torch.bool)
    selected[0, :2] = True
    weights = torch.zeros(1, 9)
    weights[0, 0] = 0.25
    weights[0, 1] = 0.75
    output = conversion(field, selected_experts=selected, routing_weights=weights)
    first, second = geometry.expert_apertures[:2]
    first_crop = output[0, first.y0:first.y1, first.x0:first.x1].real
    second_crop = output[0, second.y0:second.y1, second.x0:second.x1].real
    assert torch.allclose(second_crop, 3.0 * first_crop, rtol=1e-5, atol=1e-6)
    for aperture in geometry.expert_apertures[2:]:
        assert torch.count_nonzero(output[0, aperture.y0:aperture.y1, aperture.x0:aperture.x1]) == 0


def test_phase_dropout_is_configurable_and_training_only() -> None:
    layer = PhaseLayer(8, dropout_mode="block_phase_bypass", dropout_p=0.5, dropout_block_size=4)
    field = torch.ones(2, 8, 8, dtype=torch.complex64)
    layer.train(); layer.set_dropout_active(True)
    output = layer(field)
    assert output.shape == field.shape
    layer.eval()
    assert torch.allclose(layer(field), field * torch.exp(1j * layer.phase()).to(torch.complex64))


def test_rgb_dataset_does_not_convert_to_grayscale() -> None:
    class Fake:
        targets = [0]
        def __getitem__(self, _index):
            return Image.new("L", (32, 32), 128), 0
        def __len__(self):
            return 1
    image, label = RGBDataset(Fake())[0]
    assert image.mode == "RGB" and label == 0


def test_complete_optical_moe_forward_and_backward() -> None:
    settings = load_settings(CONFIG)
    model = VisionHomogeneousMoESurrogate(8, settings)
    output = model(torch.randn(4, 8), cu_seqlens=torch.tensor([0, 4], dtype=torch.int32))
    assert output.shape == (4, 8)
    assert model.last_detector_intensity.shape == (1, 480, 480)
    assert model.last_detector_readout.shape == (1, 120, 120)
    assert torch.all(model.last_detector_intensity >= 0)
    assert torch.all(model.last_detector_readout >= 0)
    (output.float().square().mean() + 0.1 * model.router_losses()[0]).backward()
    assert model.input_adapter.weight.grad is not None
    assert model.expert_layers[0].experts[0].raw_phase.grad is not None
    assert model.global_phase.phase.raw_phase.grad is not None
    assert model.output_adapter.weight.grad is not None
    assert model.prompt.router.gate.weight.grad is not None


def test_packed_batch_is_split_into_independent_optical_fields() -> None:
    settings = load_settings(CONFIG)
    model = VisionHomogeneousMoESurrogate(8, settings)
    hidden = torch.randn(7, 8)
    output = model(hidden, cu_seqlens=torch.tensor([0, 3, 7], dtype=torch.int32))
    assert output.shape == hidden.shape
    assert model.last_token_counts == [3, 4]
    assert model.last_input_fields.shape == (2, 120, 120)
    assert model.last_detector_intensity.shape == (2, 480, 480)
    assert model.last_routing["weights"].shape == (2, 9)
    assert torch.allclose(model.last_routing["weights"].sum(dim=1), torch.ones(2), atol=1e-5)


def test_debug_example_writes_optical_routing_and_hidden_artifacts(tmp_path) -> None:
    weights = torch.tensor([[0.4, 0.0, 0.3, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0]])
    routing = {
        "weights": weights,
        "probabilities": torch.softmax(torch.arange(9).float().reshape(1, 9), dim=1),
        "selected_mask": weights > 0,
        "prompt_amplitude": torch.rand(1, 12, 12),
    }
    complex_field = torch.complex(torch.rand(1, 12, 12), torch.rand(1, 12, 12))
    stages = [{"before_oeo": complex_field, "after_oeo": complex_field.real.to(torch.complex64)}]
    save_debug_example(tmp_path, Image.new("RGB", (8, 8)), 2, 1, [str(i) for i in range(10)],
                       torch.randn(10), torch.rand(4, 4), routing, complex_field, stages,
                       torch.rand(12, 12), torch.randn(3, 8), torch.randn(3, 8), epoch=10)
    for name in ("input_rgb.png", "prompt_expert_amplitude.png", "routing_weights.png",
                 "stage_01_before_oeo_intensity.png", "final_detector_intensity.png",
                 "hidden_comparison.png", "hidden_layernorm_comparison.png", "metadata.json"):
        assert (tmp_path / name).is_file()
