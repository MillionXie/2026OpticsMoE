from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch
from torch import nn

from experiments.qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa import training
from experiments.qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa.phase_only import (
    ADAPTATION_SCOPE,
    PANEL_FORMAT,
    PhaseOnlySettings,
    _head_gradient_comparison,
    _match_noft_identity,
    _noft_identity_candidates,
    add_phase_only_arguments,
    load_phase_only_settings,
    load_phase_only_settings_from_args,
    parameter_family_report,
    phase_only_runtime,
    phase_only_trainable_groups,
    run_phase_only,
    set_phase_only_backbone_trainable,
)
from experiments.qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa.settings import (
    IMPLEMENTATION_FILES,
    REPO_ROOT,
    implementation_sha256,
)


CONFIG = (
    REPO_ROOT
    / "experiments/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa"
    / "configs/phase_only_50e.yaml"
)


class _Stage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.raw_phase = nn.Parameter(torch.zeros(3, 224, 224))
        self.norm = nn.LayerNorm(2)
        self.mixer = nn.Linear(2, 2)
        self.fusion_gate = nn.Parameter(torch.tensor(0.6))


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(2, 2)
        self.adapter = nn.Linear(2, 2)
        self.stages = nn.ModuleList([_Stage() for _ in range(8)])

    def phase_parameters(self):
        for stage in self.stages:
            yield stage.raw_phase

    def adapter_parameters(self):
        yield from self.adapter.parameters()

    def residual_parameters(self):
        phase_ids = {id(parameter) for parameter in self.phase_parameters()}
        for stage in self.stages:
            for parameter in stage.parameters():
                if id(parameter) not in phase_ids:
                    yield parameter


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _Backbone()
        self.head = nn.Linear(2, 3)

    def phase_parameters(self):
        yield from self.backbone.phase_parameters()

    def adapter_parameters(self):
        yield from self.backbone.adapter_parameters()

    def residual_parameters(self):
        yield from self.backbone.residual_parameters()

    def head_parameters(self):
        yield from self.head.parameters()


def test_head_gradient_audit_accepts_cuda_scale_repeat_noise() -> None:
    exact = (torch.linspace(-0.5, 0.5, 257), torch.tensor([0.2, -0.1]))
    candidate = (
        exact[0] + 2.0e-5 * torch.sin(torch.arange(257, dtype=torch.float32)),
        exact[1] + torch.tensor([1.0e-6, -1.0e-6]),
    )
    rows, summary = _head_gradient_comparison(exact, candidate)
    assert summary["all_passed"] is True
    assert all(row["passed"] for row in rows)
    assert summary["maximum_absolute_difference"] < 1.0e-4


def test_head_gradient_audit_rejects_direction_or_scale_change() -> None:
    exact = (torch.linspace(-0.5, 0.5, 257),)
    candidate = (-exact[0],)
    rows, summary = _head_gradient_comparison(exact, candidate)
    assert summary["all_passed"] is False
    assert rows[0]["passed"] is False
    assert summary["minimum_cosine"] < 0.0


def test_panel_config_is_explicit_isolated_and_identity_bearing() -> None:
    settings = load_phase_only_settings(
        CONFIG, task="isic2016", method="fa_random", seed=2028
    )
    assert isinstance(settings, PhaseOnlySettings)
    assert settings.protocol.adaptation_scope == ADAPTATION_SCOPE
    assert settings.paths.output_root.name == "p12_phase_only_fa_50e"
    assert settings.paths.output_root != settings.base.paths.output_root.parent / "p12_downstream_fa_50e"
    payload = settings.to_dict()["phase_only_panel"]
    assert payload["format"] == PANEL_FORMAT
    assert payload["effective_scope"] == "phase_and_head"
    assert payload["task_head_gradient"] == "exact_bp"
    assert len(payload["panel_implementation_sha256"]) == 64


def test_cli_exposes_and_preserves_locked_scope() -> None:
    parser = add_phase_only_arguments(argparse.ArgumentParser())
    args = parser.parse_args(
        [
            "--config",
            str(CONFIG),
            "--task",
            "caltech101",
            "--method",
            "bp",
            "--seed",
            "7",
            "--adaptation-scope",
            ADAPTATION_SCOPE,
            "--max-train-batches",
            "1",
        ]
    )
    settings = load_phase_only_settings_from_args(args)
    assert settings.task == "caltech101"
    assert settings.method == "bp"
    assert settings.seed == 7
    assert settings.limits.max_train_batches == 1


def test_panel_rejects_base_config_and_scope_mismatch() -> None:
    base = CONFIG.with_name("base_50e.yaml")
    with pytest.raises(ValueError, match="phase-only config format"):
        load_phase_only_settings(base)
    with pytest.raises(ValueError, match="CLI adaptation scope"):
        load_phase_only_settings(CONFIG, adaptation_scope="joint")


def test_direct_adaptation_requires_panel_identified_noft_result(tmp_path: Path) -> None:
    settings = load_phase_only_settings(
        CONFIG,
        task="caltech101",
        method="bp",
        seed=2026,
        output_root=tmp_path / "isolated",
    )
    with pytest.raises(RuntimeError, match="own completed head-only result"):
        run_phase_only(settings)


def test_noft_identity_allows_only_exact_current_or_known_numeric_audit(
    tmp_path: Path,
) -> None:
    settings = load_phase_only_settings(
        CONFIG,
        task="isic2016",
        method="noft",
        seed=2026,
        output_root=tmp_path / "isolated",
    )
    current, legacy = _noft_identity_candidates(settings)
    current_result = {key: value for key, value in current.items() if key != "identity_version"}
    legacy_result = {key: value for key, value in legacy.items() if key != "identity_version"}
    assert _match_noft_identity(current_result, settings) == "current"
    assert _match_noft_identity(legacy_result, settings) == "legacy_numeric_audit_v1"

    legacy_result["seed"] = 2027
    with pytest.raises(RuntimeError, match="identity mismatch"):
        _match_noft_identity(legacy_result, settings)


@pytest.mark.parametrize(
    ("method", "expected_groups", "expected_phase_trainable"),
    [
        ("noft", ["head"], 0),
        ("bp", ["phase", "head"], 8),
        ("fa_pretrained", ["phase", "head"], 8),
        ("fa_random", ["phase", "head"], 8),
    ],
)
def test_parameter_families_and_optimizer_are_strict(
    tmp_path: Path,
    method: str,
    expected_groups: list[str],
    expected_phase_trainable: int,
) -> None:
    settings = load_phase_only_settings(
        CONFIG,
        task="caltech101",
        method=method,
        output_root=tmp_path / "isolated",
    )
    model = _Model()
    set_phase_only_backbone_trainable(model, settings.updates_backbone)
    report = parameter_family_report(model)  # type: ignore[arg-type]
    assert report["phase"]["trainable_tensor_count"] == expected_phase_trainable
    assert report["adapter"]["trainable_tensor_count"] == 0
    assert report["residual"]["trainable_tensor_count"] == 0
    assert report["stem"]["trainable_tensor_count"] == 0
    assert report["head"]["trainable_tensor_count"] == 2
    groups = phase_only_trainable_groups(model, settings)  # type: ignore[arg-type]
    assert [group["group_name"] for group in groups] == expected_groups
    optimizer_ids = {id(value) for group in groups for value in group["params"]}
    electronic_ids = {
        id(value)
        for value in (*model.adapter_parameters(), *model.residual_parameters())
    }
    assert optimizer_ids.isdisjoint(electronic_ids)


def test_runtime_patch_is_process_local_and_holds_frozen_backbone_in_eval() -> None:
    original_groups = training._trainable_groups
    original_set = training.P11DownstreamModel.set_backbone_trainable
    original_train = training.P11DownstreamModel.train
    with phase_only_runtime():
        assert training._trainable_groups is phase_only_trainable_groups
        assert training.P11DownstreamModel.set_backbone_trainable is not original_set
        assert training.P11DownstreamModel.train is not original_train
    assert training._trainable_groups is original_groups
    assert training.P11DownstreamModel.set_backbone_trainable is original_set
    assert training.P11DownstreamModel.train is original_train


def test_add_on_does_not_enter_or_change_locked_base_digest() -> None:
    before = implementation_sha256()
    assert all("phase_only" not in relative for relative in IMPLEMENTATION_FILES)
    load_phase_only_settings(CONFIG)
    assert implementation_sha256() == before
