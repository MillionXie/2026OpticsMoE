from __future__ import annotations

import torch

from ..hardware_contract import OPTICAL_PASSES, forward_hardware
from ..modeling import LGVQSingleMetricOEO16
from .test_model_and_training import _inputs, _small_settings


def test_simulated_hardware_contract_matches_model_forward(tmp_path) -> None:
    settings = _small_settings(tmp_path, target_name="spatial")
    torch.manual_seed(23)
    model = LGVQSingleMetricOEO16(settings).eval()
    vision, quality, language, mask = _inputs(frame_count=settings.frame_count)
    batch = {
        "vision_tokens": vision,
        "quality_tokens": quality,
        "language_tokens": language,
        "language_mask": mask,
    }
    with torch.no_grad():
        expected = model(vision, quality, language, mask, optical_enabled=True)
        actual = forward_hardware(model, batch, ["a", "b"])
    assert tuple(actual.amplitudes) == OPTICAL_PASSES
    assert torch.allclose(actual.prediction, expected["prediction"], atol=1e-6, rtol=1e-6)


def test_each_stop_point_returns_one_more_playable_amplitude(tmp_path) -> None:
    settings = _small_settings(tmp_path, target_name="spatial")
    model = LGVQSingleMetricOEO16(settings).eval()
    vision, quality, language, mask = _inputs(batch=1, frame_count=settings.frame_count)
    batch = {
        "vision_tokens": vision,
        "quality_tokens": quality,
        "language_tokens": language,
        "language_mask": mask,
    }
    with torch.no_grad():
        for index, optical_pass in enumerate(OPTICAL_PASSES, 1):
            result = forward_hardware(model, batch, ["sample"], stop_before=optical_pass)
            assert len(result.amplitudes) == index
            assert result.amplitudes[optical_pass].shape == (1, 88, 88)
            assert result.prediction is None

