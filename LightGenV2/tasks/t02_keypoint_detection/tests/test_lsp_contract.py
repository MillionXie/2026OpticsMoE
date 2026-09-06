"""Contract tests for the LightGenV2 LSP task."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from LightGenV2.tasks.t01_object_retrieval.modeling import DensePhasePlane
from LightGenV2.tasks.t02_keypoint_detection.settings import load_settings
from LightGenV2.tasks.t02_keypoint_detection.visualize import render


TASK_DIR = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("filename", "variant"),
    (
        ("moe_optical_router_scale_matched_dc20.yaml", "optical_router_scale_matched_moe"),
        ("moe_optical_router_scale_matched_dc20_no_shift.yaml", "optical_router_scale_matched_moe"),
        ("d2nn_active_expert_matched_dc20.yaml", "d2nn_active_expert_matched"),
    ),
)
def test_formal_profiles_share_dc_robust_contract(filename: str, variant: str) -> None:
    settings = load_settings(TASK_DIR / "configs" / filename)
    assert settings.lightgen_model_variant == variant
    assert settings.router_backend == "optical"
    assert settings.top_k == 2
    assert settings.fusion_mode == "scale_matched_convex"
    assert settings.language_optical_zero_order_enabled is True
    assert settings.language_optical_amplitude_zero_order_intensity_min == pytest.approx(0.20)
    assert settings.language_optical_phase_zero_order_intensity_max == pytest.approx(0.30)
    assert settings.language_optical_ccd_noise_distribution == "truncated_biased_gaussian"
    assert settings.language_optical_k_space_enabled is True
    assert settings.language_optical_distance_m == pytest.approx(0.10)
    assert settings.language_optical_pixel_pitch_um == pytest.approx(17.0)
    assert settings.periodic_test_interval_epochs == 5


def test_d2nn_phase_budget_matches_top2_experts() -> None:
    settings = load_settings(TASK_DIR / "configs" / "d2nn_active_expert_matched_dc20.yaml")
    assert settings.d2nn_phase_layers * settings.d2nn_phase_size**2 == settings.top_k * settings.expert_size**2


def test_no_shift_ablation_only_disables_pixel_translations() -> None:
    settings = load_settings(
        TASK_DIR / "configs" / "moe_optical_router_scale_matched_dc20_no_shift.yaml"
    )
    assert settings.language_optical_max_shift_pixels == 0
    assert settings.language_optical_phase_shift_pixels == 0
    assert settings.language_optical_ccd_shift_pixels == 0
    assert settings.optical_router_input_shift_pixels == 0
    assert settings.optical_router_phase_shift_pixels == 0
    assert settings.optical_router_ccd_shift_pixels == 0
    assert settings.language_optical_zero_order_enabled is True
    assert settings.language_optical_amplitude_zero_order_intensity_min == pytest.approx(0.20)
    assert settings.language_optical_ccd_noise_distribution == "truncated_biased_gaussian"


def test_visualizer_reads_canonical_lsp_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best_checkpoint.pt"
    torch.save({
        "epoch": 10,
        "weight_variant": "ema",
        "core": {
            "hybrid.optical_branch.phase1.raw_phase": torch.zeros(8, 8),
            "hybrid.optical_branch.phase2.raw_phase": torch.ones(8, 8),
        },
    }, checkpoint)
    report = render(checkpoint, tmp_path / "visualization")
    assert report["plane_count"] == 2
    assert (tmp_path / "visualization" / "best_phase_overview.png").is_file()
    assert (tmp_path / "visualization" / "phase_statistics.json").is_file()


def test_dense_phase_uses_two_pi_sigmoid() -> None:
    settings = SimpleNamespace(
        language_optical_phase_init_std=0.1,
        language_optical_phase_dropout_p=0.0,
        language_optical_phase_dropout_block_size=8,
    )
    phase = DensePhasePlane(12, settings).physical_phase()
    assert torch.all((phase > 0) & (phase < 2 * torch.pi))
