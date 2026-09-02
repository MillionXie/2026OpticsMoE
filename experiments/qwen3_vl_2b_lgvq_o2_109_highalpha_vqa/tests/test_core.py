from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import torch

from experiments.qwen3_vl_2b_lgvq_o2_109_highalpha_vqa.cache_qwen_inputs import (
    MERGED_TOKENS,
    VISION_OUTPUT_SIZE,
    _main_merger_features,
)
from experiments.qwen3_vl_2b_lgvq_o2_109_highalpha_vqa.metrics import (
    _kendall_tau_b,
)
from experiments.qwen3_vl_2b_lgvq_o2_109_highalpha_vqa.modeling import (
    ScaleMatchedFusion,
    _phase,
    _phase_to_raw,
    _resize_wrapped_phase,
)
from experiments.qwen3_vl_2b_lgvq_o2_109_highalpha_vqa.run import (
    synthetic_smoke,
)
from experiments.qwen3_vl_2b_lgvq_o2_109_highalpha_vqa.settings import (
    ALLOWED_ROUTERS,
    load_settings,
)
from experiments.qwen3_vl_2b_lgvq_o2_109_highalpha_vqa.training import (
    meets_reference_optical_500,
)


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "configs" / "release"


def test_release_configs_are_strict_optical_top2_hardware109() -> None:
    expected_alpha = {
        "alpha20.yaml": (0.20, 0.90, 0.30),
        "alpha35.yaml": (0.35, 0.90, 0.42),
        "alpha50.yaml": (0.50, 0.90, 0.57),
        "alpha65.yaml": (0.65, 0.90, 0.70),
    }
    assert {path.name for path in RELEASE.glob("*.yaml")} == {
        "common.yaml",
        *expected_alpha,
    }
    assert ALLOWED_ROUTERS == {"optical"}

    for name, (minimum, maximum, initial) in expected_alpha.items():
        settings = load_settings(RELEASE / name)
        assert settings.router_backend == "optical"
        assert settings.top_k == 2
        assert settings.router_weight_normalization == "power_l2"
        assert settings.router_straight_through
        assert settings.geometry.canvas_size == 518
        assert settings.geometry.active_size == 478
        assert settings.geometry.quadrant_size == 232
        assert settings.geometry.expert_size == 109
        assert settings.geometry.expert_pitch == 123
        assert settings.geometry.expert_pitch - settings.geometry.expert_size == 14
        assert settings.fusion_alpha_min == pytest.approx(minimum)
        assert settings.fusion_alpha_max == pytest.approx(maximum)
        assert settings.fusion_alpha_initial == pytest.approx(initial)
        assert settings.fusion_alpha_min >= 0.20
        assert settings.feature_contract == (
            "qwen3vl_full_visual_main_merger_196x2048_v1"
        )
        assert settings.input_width == 2048


def test_teacher_reference_requires_all_ten_metrics() -> None:
    metrics = {
        "spatial": {
            "srcc": 0.6710,
            "krcc": 0.4909,
            "plcc": 0.7106,
            "rmse": 8.197,
            "mae": 6.493,
        },
        "temporal": {
            "srcc": 0.8604,
            "krcc": 0.6623,
            "plcc": 0.8784,
            "rmse": 6.721,
            "mae": 5.144,
        },
    }
    assert meets_reference_optical_500(metrics)
    metrics["temporal"]["rmse"] = 6.722
    assert not meets_reference_optical_500(metrics)


def _masked_rms(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weighted = value.float().square() * mask.unsqueeze(-1)
    denominator = (mask.sum(1).float() * value.shape[-1]).clamp_min(1.0)
    return (weighted.sum((1, 2)) / denominator).sqrt()


def test_scale_matched_fusion_preserves_rms_and_hard_alpha_range() -> None:
    settings = load_settings(RELEASE / "alpha35.yaml", synthetic=True)
    fusion = ScaleMatchedFusion(settings)
    generator = torch.Generator().manual_seed(7)
    electronic = torch.randn(3, 6, 11, generator=generator) * 2.5
    optical = torch.randn(3, 6, 11, generator=generator) * 70.0 + 13.0
    mask = torch.tensor(
        [
            [True, True, True, False, False, False],
            [True, False, True, False, True, False],
            [True, True, False, True, False, True],
        ]
    )

    fused = fusion(electronic, optical, mask)
    torch.testing.assert_close(
        _masked_rms(fused, mask),
        _masked_rms(electronic, mask),
        rtol=2.0e-5,
        atol=2.0e-5,
    )
    assert torch.count_nonzero(fused.masked_select(~mask.unsqueeze(-1))) == 0
    assert float(fusion.alpha.detach()) == pytest.approx(
        settings.fusion_alpha_initial
    )
    assert fusion.last_diagnostics["fused_to_electronic_rms"] == pytest.approx(
        1.0, abs=2.0e-5
    )

    with torch.no_grad():
        fusion.raw_alpha.fill_(-100.0)
    assert float(fusion.alpha.detach()) == pytest.approx(
        settings.fusion_alpha_min, abs=1.0e-6
    )
    with torch.no_grad():
        fusion.raw_alpha.fill_(100.0)
    assert float(fusion.alpha.detach()) == pytest.approx(
        settings.fusion_alpha_max, abs=1.0e-6
    )


def test_kendall_tau_b_includes_tie_correction() -> None:
    left = torch.tensor([1.0, 1.0, 2.0, 3.0])
    right = torch.tensor([1.0, 2.0, 2.0, 3.0])
    # Four concordant pairs, one x-only tie and one y-only tie:
    # tau_b = 4 / sqrt((4+1)*(4+1)) = 0.8.
    assert _kendall_tau_b(left, right) == pytest.approx(0.8, abs=1.0e-12)
    assert _kendall_tau_b(right, left) == pytest.approx(0.8, abs=1.0e-12)
    assert _kendall_tau_b(left, left) == pytest.approx(1.0, abs=1.0e-12)
    assert math.isnan(_kendall_tau_b(torch.ones(4), torch.arange(4.0)))


class _TupleVision:
    def __init__(self, outputs: tuple[Any, ...]) -> None:
        self.outputs = outputs
        self.calls = 0

    def __call__(
        self,
        pixel_values: torch.Tensor,
        *,
        grid_thw: torch.Tensor,
    ) -> tuple[Any, ...]:
        assert pixel_values.numel() > 0
        assert tuple(grid_thw.shape) == (1, 3)
        self.calls += 1
        return self.outputs


@pytest.mark.parametrize("include_empty_deepstack_slot", [False, True])
def test_main_merger_accepts_qwen_tuple_return_without_deepstack(
    include_empty_deepstack_slot: bool,
) -> None:
    merged = torch.arange(
        MERGED_TOKENS * VISION_OUTPUT_SIZE, dtype=torch.float32
    ).reshape(MERGED_TOKENS, VISION_OUTPUT_SIZE)
    outputs: tuple[Any, ...] = (merged,)
    if include_empty_deepstack_slot:
        outputs = (*outputs, [])
    visual = _TupleVision(outputs)

    result = _main_merger_features(
        visual,
        torch.ones(1),
        torch.tensor([[1, 28, 28]]),
        image_count=1,
    )
    assert visual.calls == 1
    assert result.shape == (1, MERGED_TOKENS, VISION_OUTPUT_SIZE)
    torch.testing.assert_close(result[0], merged)


def test_main_merger_rejects_tuple_return_containing_deepstack() -> None:
    merged = torch.zeros(MERGED_TOKENS, VISION_OUTPUT_SIZE)
    visual = _TupleVision((merged, [torch.ones(1)]))
    with pytest.raises(RuntimeError, match="DeepStack"):
        _main_merger_features(
            visual,
            torch.ones(1),
            torch.tensor([[1, 28, 28]]),
            image_count=1,
        )


def test_wrapped_phase_resize_preserves_circular_neighborhood() -> None:
    source_phase = torch.tensor(
        [
            [
                [0.04, 2.0 * math.pi - 0.04],
                [2.0 * math.pi - 0.06, 0.06],
            ],
            [
                [2.0 * math.pi - 0.03, 0.03],
                [0.05, 2.0 * math.pi - 0.05],
            ],
        ],
        dtype=torch.float32,
    )
    source_raw = _phase_to_raw(source_phase)
    target = torch.empty(2, 9, 11, dtype=torch.float64)
    resized_raw = _resize_wrapped_phase(source_raw, target)

    assert resized_raw.shape == target.shape
    assert resized_raw.dtype == target.dtype
    assert bool(torch.isfinite(resized_raw).all())
    resized_phase = _phase(resized_raw.float())
    circular_distance_to_zero = torch.minimum(
        resized_phase, 2.0 * math.pi - resized_phase
    )
    # Circular cos/sin interpolation must stay close to the 0/2pi boundary;
    # direct scalar interpolation would incorrectly cross through pi.
    assert float(circular_distance_to_zero.max()) < 0.15
    assert float(circular_distance_to_zero[:, 4, 5].max()) < 0.10


def test_synthetic_smoke_runs_only_optical_top2() -> None:
    settings = load_settings(RELEASE / "alpha20.yaml", synthetic=True)
    report = synthetic_smoke(settings)
    assert report["status"] == "passed"
    assert report["targets"] == ["spatial", "temporal"]
    assert report["alignment_output"] is False
    assert set(report["variants"]) == {"optical"}

    optical = report["variants"]["optical"]
    assert optical["prediction_shape"] == [2, 2]
    assert optical["parameters"]["router"] > 0
    assert set(optical["fusion"]) == {
        "vision_expert",
        "vision_global",
        "language_expert",
        "language_global",
    }
    for diagnostics in optical["fusion"].values():
        assert settings.fusion_alpha_min <= diagnostics["alpha"]
        assert diagnostics["alpha"] <= settings.fusion_alpha_max
