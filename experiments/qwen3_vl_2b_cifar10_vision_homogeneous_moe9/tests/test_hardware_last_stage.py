from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from experiments.qwen3_vl_2b_cifar10_vision_homogeneous_moe9.hardware_last_stage.bmp import export_plane_bmp
from experiments.qwen3_vl_2b_cifar10_vision_homogeneous_moe9.hardware_last_stage.ccd_readout import CCDReadoutModel
from experiments.qwen3_vl_2b_cifar10_vision_homogeneous_moe9.optics.moe import (
    FullPlaneDetectorReadout,
    VisionHomogeneousMoESurrogate,
)


def _settings():
    return SimpleNamespace(
        canvas_size=480,
        detector_pool_kernel=4,
        detector_layernorm_eps=1e-5,
        detector_nonlinearity="relu",
        input_adapter_dim=120,
        head_type="normalized_linear",
    )


def test_detector_direct_intensity_matches_complex_field() -> None:
    readout = FullPlaneDetectorReadout(_settings())
    amplitude = torch.rand(2, 480, 480)
    from_field, intensity = readout(torch.complex(amplitude, torch.zeros_like(amplitude)))
    from_ccd = readout.forward_intensity(intensity)
    assert from_ccd.shape == (2, 120, 120)
    assert torch.allclose(from_field, from_ccd)
    assert torch.all(intensity >= 0)


def test_ccd_readout_model_shape_and_gradients() -> None:
    model = CCDReadoutModel(_settings(), hidden_size=1024, num_classes=10)
    logits = model(torch.rand(3, 480, 480), torch.tensor([100, 110, 120]))
    assert logits.shape == (3, 10)
    logits.sum().backward()
    assert model.output_adapter.weight.grad is not None
    assert model.head.classifier.weight.grad is not None


def test_hardware_bmp_dimensions_and_centering(tmp_path: Path) -> None:
    amplitude = torch.ones(450, 450)
    amp_path = tmp_path / "amplitude.bmp"
    phase_path = tmp_path / "phase.bmp"
    amp = export_plane_bmp(amplitude, amp_path, "amplitude", 2, 1920, 1080)
    phase = export_plane_bmp(torch.zeros_like(amplitude), phase_path, "phase", 2, 1920, 1200)
    assert Image.open(amp_path).size == (1920, 1080)
    assert Image.open(phase_path).size == (1920, 1200)
    assert amp["active_bounds_xyxy"] == [510, 90, 1410, 990]
    assert phase["active_bounds_xyxy"] == [510, 150, 1410, 1050]


def test_final_oeo_and_global_phase_are_adjacent_in_forward() -> None:
    source = inspect.getsource(VisionHomogeneousMoESurrogate.forward)
    final_oeo = source.index("field = self.interlayer_conversions[index]")
    global_phase = source.index("detector_field = self.to_detector(self.global_phase(field))")
    between = source[final_oeo:global_phase]
    assert "self.to_detector" not in between
    assert "propagation(" not in between

