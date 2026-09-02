from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.train_optical_retrieval import (
    save_checkpoint,
)

from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.optical_blocks import (
    LanguageTwoBlockOpticalCore,
    VisionTwoBlockOpticalCore,
)

from ..modeling import (
    BalancedLanguageCore,
    BalancedVisionCore,
    _ScaleMatchedFusionMixin,
    balanced_checkpoint_architecture,
    _range_gate,
    _range_logit,
)
from ..checkpoint_report import build_report
from ..settings import load_settings


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs" / "release"


class _FusionHarness(_ScaleMatchedFusionMixin):
    def __init__(self) -> None:
        self.fusion_mode = "scale_matched_convex"
        self.fusion_alpha_min = 0.05
        self.fusion_alpha_max = 0.49
        self.fusion_rms_epsilon = 1.0e-6
        self.fusion_ablation_mode = "none"
        self.last_fusion_diagnostics = {}


def _masked_rms(value: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    valid = (~padding_mask).unsqueeze(-1).to(value.dtype)
    count = (~padding_mask).sum(1).to(value.dtype) * value.shape[-1]
    return ((value.square() * valid).sum((1, 2)) / count).sqrt()


def test_release_variants_share_protocol_and_have_disjoint_alpha_ranges() -> None:
    free = load_settings(CONFIG_ROOT / "alpha_free.yaml")
    low = load_settings(CONFIG_ROOT / "alpha_low_lt_0p5.yaml")
    high = load_settings(CONFIG_ROOT / "alpha_high_gt_0p5.yaml")
    electronic = load_settings(CONFIG_ROOT / "electronic_only.yaml")
    assert (low.fusion_alpha_min, low.fusion_alpha_max, low.fusion_alpha_initial) == (
        0.05,
        0.49,
        0.055,
    )
    assert (high.fusion_alpha_min, high.fusion_alpha_max, high.fusion_alpha_initial) == (
        0.51,
        0.95,
        0.55,
    )
    assert free.fusion_alpha_min < 0.5 < free.fusion_alpha_max
    assert electronic.fusion_mode == "electronic_only"
    for settings in (free, low, high, electronic):
        assert settings.epochs == 30
        assert settings.optimizer_steps_per_epoch is None
        assert settings.evaluate_test_each_epoch is True
        assert settings.test_evaluation_interval_epochs == 5
        assert settings.fusion_detach_scale_statistics is True
        assert settings.initialization_sha256 == (
            "6a27f54d8c869cce46150583383a127b0ba47b3d34503f5753aa23974ac1e55d"
        )


def test_gate_is_strictly_below_or_above_half_and_has_gradient() -> None:
    low_raw = _range_logit(0.055, 0.05, 0.49).requires_grad_(True)
    low = _range_gate(low_raw, 0.05, 0.49)
    torch.testing.assert_close(low, torch.tensor(0.055), atol=1e-7, rtol=0)
    assert 0.05 < float(low.detach()) < 0.49
    low.backward()
    assert low_raw.grad is not None and low_raw.grad > 0
    high_values = _range_gate(torch.tensor([-100.0, 0.0, 100.0]), 0.51, 0.95)
    assert torch.all(high_values >= 0.51)
    assert torch.all(high_values <= 0.95)


def test_warmstart_core_state_contract_is_exactly_compatible() -> None:
    settings = load_settings(CONFIG_ROOT / "alpha_low_lt_0p5.yaml")
    source_and_target = (
        (
            VisionTwoBlockOpticalCore(16, 16, settings),
            BalancedVisionCore(16, 16, settings),
        ),
        (
            LanguageTwoBlockOpticalCore(16, 16, settings),
            BalancedLanguageCore(16, 16, settings),
        ),
    )
    for source, target in source_and_target:
        source_state = source.state_dict()
        target_state = target.state_dict()
        assert set(source_state) == set(target_state)
        assert all(
            source_state[key].shape == target_state[key].shape
            for key in source_state
        )


def test_scale_matching_uses_all_valid_tokens_preserves_token_contrast() -> None:
    harness = _FusionHarness()
    padding = torch.tensor([[False, False, True], [False, True, True]])
    electronic = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [5.0, 1.0, 2.0, 1.0], [999.0] * 4],
            [[2.0, 3.0, 1.0, 4.0], [888.0] * 4, [777.0] * 4],
        ]
    )
    optical = torch.tensor(
        [
            [[20.0, 10.0, 5.0, 2.0], [1.0, 4.0, 3.0, 9.0], [9999.0] * 4],
            [[8.0, 1.0, 5.0, 2.0], [8888.0] * 4, [7777.0] * 4],
        ]
    )
    fused = harness._fuse(
        electronic, optical, torch.tensor(0.30), padding, "block1"
    )
    torch.testing.assert_close(
        _masked_rms(fused, padding),
        _masked_rms(electronic, padding),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.count_nonzero(fused[padding]) == 0
    diagnostics = harness.fusion_diagnostics()["block1"]
    assert abs(diagnostics["post_optical_to_electronic_rms_ratio"] - 1.0) < 1e-7
    assert abs(diagnostics["fused_to_electronic_rms_ratio"] - 1.0) < 1e-5
    # The first sample's two valid tokens retain different norms; this is not
    # per-token normalization.
    assert not torch.isclose(fused[0, 0].norm(), fused[0, 1].norm())


def test_alpha_zero_is_electronic_identity_and_optical_off_skips_optical() -> None:
    harness = _FusionHarness()
    padding = torch.tensor([[False, False]])
    electronic = torch.randn(1, 2, 7)
    optical = torch.randn(1, 2, 7) * 19.0
    output = harness._fuse(
        electronic, optical, torch.tensor(0.0), padding, "alpha_zero"
    )
    torch.testing.assert_close(output, electronic, atol=2e-6, rtol=2e-6)
    harness.set_fusion_ablation("remove_optical")
    output = harness._fuse(
        electronic, None, torch.tensor(0.8), padding, "optical_off"
    )
    assert torch.equal(output, electronic)


def test_alpha_zero_identity_survives_rms_epsilon_floor() -> None:
    harness = _FusionHarness()
    padding = torch.tensor([[False, False]])
    electronic = torch.full((1, 2, 7), 1.0e-14)
    optical = torch.randn(1, 2, 7)
    output = harness._fuse(
        electronic, optical, torch.tensor(0.0), padding, "alpha_zero_tiny"
    )
    assert torch.equal(output, electronic)


def test_remove_electronic_reports_actual_zero_one_coefficients() -> None:
    harness = _FusionHarness()
    harness.set_fusion_ablation("remove_electronic")
    padding = torch.tensor([[False, False]])
    output = harness._fuse(
        torch.randn(1, 2, 7),
        torch.randn(1, 2, 7),
        torch.tensor(0.30),
        padding,
        "optical_only",
    )
    assert torch.isfinite(output).all()
    diagnostics = harness.fusion_diagnostics()["optical_only"]
    assert diagnostics["electronic_coefficient"] == 0.0
    assert diagnostics["optical_coefficient"] == 1.0

    # The alpha=0 endpoint identity belongs only to the full fusion equation;
    # an explicit remove-electronic counterfactual must remain optical-only.
    second_electronic = torch.randn(1, 2, 7)
    second_optical = torch.randn(1, 2, 7)
    optical_only = harness._fuse(
        second_electronic,
        second_optical,
        torch.tensor(0.0),
        padding,
        "optical_only_alpha_zero",
    )
    assert not torch.equal(optical_only, second_electronic)


def test_optical_off_skips_all_four_formal_and_two_hardware_calls() -> None:
    def forbidden_optical_call(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Optical propagation/CCD decoding must be skipped")

    for config_name, ablation in (
        ("electronic_only.yaml", None),
        ("alpha_low_lt_0p5.yaml", "remove_optical"),
    ):
        settings = load_settings(CONFIG_ROOT / config_name)
        vision = BalancedVisionCore(16, 16, settings)
        language = BalancedLanguageCore(16, 16, settings)
        if ablation is not None:
            vision.set_fusion_ablation(ablation)
            language.set_fusion_ablation(ablation)
        for core in (vision, language):
            core.optical_branch.run_expert_block = forbidden_optical_call
            core.optical_branch.encode_global_input = forbidden_optical_call
            core.optical_branch.run_global_block = forbidden_optical_call
            core.optical_branch.decode_measured_ccd = forbidden_optical_call

        vision.forward_groups(
            [torch.randn(4, 16)],
            causal=False,
            spatial_shapes=[(1, 2, 2)],
        )
        language_stage1, _ = language.forward_stage_groups(
            0, [torch.randn(4, 16)]
        )
        language.forward_stage_groups(1, [language_stage1])
        assert language.detector_features_from_cached(
            torch.randn(4, 192), torch.randn(8, 8)
        ).shape == (384,)
        assert language.detector_features_from_block2_inputs(
            [torch.randn(4, 192)], torch.randn(1, 8, 8)
        ).shape == (1, 384)


class _Surrogate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))
        self.core = SimpleNamespace(expert_layers=[])


class _Replacement:
    checkpoint_architecture = "unit_test_architecture"
    language_optical_layer_indexes = ()

    def __init__(self) -> None:
        self.vision_surrogate = _Surrogate()
        self.language_surrogate = _Surrogate()


def test_observed_test_checkpoint_metadata_can_state_leakage_truthfully(tmp_path: Path) -> None:
    replacement = _Replacement()
    readout = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(readout.parameters(), lr=0.1)
    settings = SimpleNamespace(
        embedding_dim=2,
        detector_output_size=2,
        instruction="test",
        model_id="test",
        expert_layers=0,
        vision_tap_stages=(),
        lambda_kd=0.0,
        lambda_relational_kd=0.0,
        lambda_ret=1.0,
        lambda_gallery=1.0,
        lambda_teacher_gallery=0.0,
        lambda_router_balance=0.0,
        lambda_router_importance=0.0,
        phase_dc_enabled=False,
        lambda_phase_dc=0.0,
        lambda_ccd_operating_point=0.0,
        phase_dc_start_epoch=1,
        temperature=0.1,
        gallery_temperature=0.1,
        gallery_prototype_stop_gradient=False,
        learning_rate=1e-4,
        adapter_learning_rate=1e-4,
        readout_learning_rate=1e-4,
        router_learning_rate=1e-4,
        phase_learning_rate=1e-4,
        phase_focus_enabled=False,
        phase_focus_warmup_epochs=0,
        phase_focus_interval_epochs=1,
        ema_decay=0.995,
    )
    path = tmp_path / "observed.pt"
    save_checkpoint(
        path,
        replacement,
        readout,
        optimizer,
        5,
        1.0,
        settings,
        weight_variant="ema",
        selection_criterion="maximum_periodically_observed_test_top1",
        test_metrics_used_for_selection=True,
    )
    metadata = torch.load(path, weights_only=False)["metadata"]
    assert metadata["selection_criterion"] == (
        "maximum_periodically_observed_test_top1"
    )
    assert metadata["test_metrics_used_for_selection"] is True


def test_checkpoint_report_exports_all_four_coefficients(tmp_path: Path) -> None:
    config = CONFIG_ROOT / "alpha_low_lt_0p5.yaml"
    raw = _range_logit(0.20, 0.05, 0.49)
    state = {
        "core.block1_optical_fusion_logit": raw,
        "core.block2_optical_fusion_logit": raw,
    }
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "epoch": 7,
            "train_loss": 1.2,
            "vision_optical": state,
            "language_optical": state,
            "metadata": {
                "weight_variant": "ema",
                "optical_architecture": balanced_checkpoint_architecture(
                    "scale_matched_convex", 0.05, 0.49, 1.0e-6
                ),
                "selection_criterion": "maximum_periodically_observed_test_top1",
                "test_metrics_used_for_selection": True,
            },
        },
        checkpoint,
    )
    report = build_report(config, checkpoint)
    assert set(report["gates"]) == {
        "vision_block1",
        "vision_block2",
        "language_block1",
        "language_block2",
    }
    for gate in report["gates"].values():
        assert abs(gate["alpha"] - 0.20) < 1e-6
        assert abs(
            gate["electronic_coefficient"] + gate["optical_coefficient"] - 1.0
        ) < 1e-7


def test_checkpoint_report_rejects_wrong_alpha_range(tmp_path: Path) -> None:
    state = {
        "core.block1_optical_fusion_logit": torch.tensor(0.0),
        "core.block2_optical_fusion_logit": torch.tensor(0.0),
    }
    checkpoint = tmp_path / "high_range_checkpoint.pt"
    torch.save(
        {
            "epoch": 1,
            "train_loss": 1.0,
            "vision_optical": state,
            "language_optical": state,
            "metadata": {
                "optical_architecture": balanced_checkpoint_architecture(
                    "scale_matched_convex", 0.51, 0.95, 1.0e-6
                )
            },
        },
        checkpoint,
    )
    try:
        build_report(CONFIG_ROOT / "alpha_low_lt_0p5.yaml", checkpoint)
    except RuntimeError as exc:
        assert "checkpoint/config architecture mismatch" in str(exc)
    else:
        raise AssertionError("Mismatched alpha ranges must fail closed")
