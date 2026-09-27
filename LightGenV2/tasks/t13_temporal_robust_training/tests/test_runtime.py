from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runtime"))
from study import GROUPS, make_config
from experiment import detector_noise_evaluation
from hardware import rasterize_phase
from settings_adapter import load_settings


@pytest.mark.parametrize("group", GROUPS)
@pytest.mark.parametrize("purpose", ("train", "deployment", "nominal"))
def test_formal_dc_settings(tmp_path, group, purpose):
    import yaml
    from LightGenV2.tasks.t06_video_quality_assessment.multivideo_settings import MultiVideoSettings
    original = MultiVideoSettings.validate
    raw = make_config(group, purpose=purpose)
    config = tmp_path / "formal.yaml"
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    result = load_settings(config)
    assert not result.synthetic
    assert MultiVideoSettings.validate is original
    expected = 0.0 if purpose == "nominal" or (purpose == "train" and not GROUPS[group]["dc"]) else 0.30
    assert result.unmodulated_power_fraction_eval == expected
    raw["optics"]["unmodulated_power_fraction_min"] = -0.1
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid unmodulated-power interval"):
        load_settings(config)
    assert MultiVideoSettings.validate is original
from LightGenV2.tasks.t06_video_quality_assessment.models import multivideo9x4 as optics


@pytest.fixture
def settings(tmp_path):
    import yaml
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(make_config("r3_ccd_dc_intrain")), encoding="utf-8")
    return load_settings(config)


def test_sensor_eval_does_not_enable_network_training(settings):
    image = torch.ones(1, 8, 8, requires_grad=True)
    original = optics._perturb_ccd
    assert torch.equal(original(image, settings, training=False), image)
    with detector_noise_evaluation(settings, scale=1):
        result = optics._perturb_ccd(image, settings, training=False)
        assert not torch.equal(result, image)
        result.sum().backward()
    assert optics._perturb_ccd is original
    assert torch.isfinite(image.grad).all()


def test_phase_raster_keeps_wrapping_and_aperture():
    phase = torch.tensor([[0.01, 6.27], [6.27, 0.01]], requires_grad=True)
    mapped = rasterize_phase(phase, model_pitch_um=17, device_pitch_um=8)
    assert mapped.shape == (4, 4)
    assert ((mapped < 0.1) | (mapped > 6.2)).all()
    mapped.sum().backward()
    assert torch.isfinite(phase.grad).all()


def test_device_mapping_has_gradient(settings):
    field = torch.complex(torch.rand(1, 4, 4), torch.rand(1, 4, 4)).requires_grad_()
    mapped = optics._resample_complex_field(field, 9)
    mapped.abs().square().sum().backward()
    assert field.grad is not None and torch.isfinite(field.grad).all()
    assert settings.propagation_active_size == 1016
    assert settings.propagation_canvas_size == 1101


def test_eval_restores_model_grid_and_mode(settings):
    from dataclasses import replace
    from experiment import install_common_selection_evaluator
    from types import SimpleNamespace
    train_settings = replace(settings, modulator_pixel_pitch_um=17)
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.optics = optics._FullFieldBase(train_settings)
    model = Tiny().train()
    old_prop = model.optics.propagation
    def callback(model, payload, settings, device, **kwargs):
        assert model.optics.settings.modulator_pixel_pitch_um == 8
        assert model.optics.propagation_size == 1101
        model.eval()
        return {"srcc": 0}
    module = SimpleNamespace(evaluate=callback)
    install_common_selection_evaluator(module, settings, seed=1)
    module.evaluate(model, {}, train_settings, torch.device("cpu"))
    assert model.training
    assert model.optics.propagation is old_prop
    assert model.optics.propagation_size == 518


def test_logical_six_stage_capture_and_measured_replay(settings):
    from dataclasses import replace
    from hardware import measured_ccd_boundary, STAGES
    from LightGenV2.tasks.t06_video_quality_assessment.multivideo_settings import MultiVideoGeometry
    g = MultiVideoGeometry(canvas_size=121, active_size=117, video_grid=4, video_count=16,
        video_tile_size=28, video_tile_pitch=29, video_tile_offset=1,
        frame_grid=2, frames_per_video=4, frame_lane_size=14, frame_lane_pitch=14,
        frame_expert_size=6, frame_expert_pitch=8, video_field_size=12,
        video_expert_pitch=16, video_phase_tile_size=26)
    small = replace(settings, geometry=g, vision_input_width=32, quality_input_width=6,
        language_input_width=40, model_width=24, detector_projection_size=8, head_width=32,
        temporal_readout_hidden_width=64, temporal_readout_rank=16, maximum_language_tokens=12,
        frame_router_intervals=((2,6),(8,12)), video_router_intervals=((4,10),(18,24)),
        input_shift_pixels=0, phase_shift_pixels=0, ccd_shift_pixels=0,
        phase_dropout_p=0, k_space_enabled=False, synthetic=True)
    model = optics.build_model(small).eval()
    inputs = (torch.randn(1,16,4,49,32), torch.randn(1,16,4,49,6),
              torch.randn(1,8,40), torch.ones(1,8,dtype=torch.bool))
    from amplitude import bounded_graph, quantize_amplitude
    with bounded_graph(), torch.inference_mode(), measured_ccd_boundary(model) as tap:
        reference = model(*inputs, optical_enabled=True)["prediction"]
    assert all(float(value.max()) <= 1.000001 for value in tap["amplitudes"].values())
    for amplitude in tap["amplitudes"].values():
        recovered = quantize_amplitude(amplitude).float() / 255
        assert float((recovered-amplitude).abs().max()) <= 0.5/255 + 1e-7
    assert tuple(tap["detectors"]) == STAGES
    assert all(value.shape == (1,121,121) for value in tap["detectors"].values())
    with bounded_graph(), torch.inference_mode(), measured_ccd_boundary(model, tap["detectors"]):
        replay = model(*inputs, optical_enabled=True)["prediction"]
    assert torch.equal(reference, replay)
    with pytest.raises(ValueError):
        with measured_ccd_boundary(model, {"language_global": torch.ones(1,121,121)}):
            pass
