from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import yaml

from experiments.qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa.settings import (
    METHODS,
    REPO_ROOT,
    TASKS,
    RunLimits,
    add_settings_arguments,
    load_settings,
    load_settings_from_args,
)


CONFIG = (
    REPO_ROOT
    / "experiments/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa"
    / "configs/base_50e.yaml"
)


def _mutated_config(tmp_path: Path, dotted_key: str, value: object) -> Path:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    target = raw
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_base_recipe_is_the_locked_50_epoch_p11_protocol() -> None:
    settings = load_settings(CONFIG)
    assert settings.task == "caltech101"
    assert settings.method == "bp"
    assert settings.seed == 2026
    assert settings.training.head_only_epochs == 50
    assert settings.training.adaptation_epochs == 50
    assert settings.run_epochs == 50
    assert settings.inherited_pipeline_epochs == 100
    assert settings.model.axis_schedule == ("token", "channel") * 4
    assert settings.model.architecture_signature == (11, 1, 2, 4)
    assert settings.model.expected_optical_parameters == 1_204_224
    assert settings.model.optical_gate_min == 0.5
    assert settings.optimizer.phase_learning_rate == pytest.approx(3.0e-3)
    assert settings.optimizer.phase_weight_decay == 0.0
    assert settings.p11_config["num_classes"] == 1000
    assert settings.p11_config["mixer_width"] == 96
    assert settings.p11_config["token_axis_propagation_distance_m"] == 0.05
    assert settings.p11_config["channel_axis_propagation_distance_m"] == 0.05
    assert settings.paths.source_backbone_sha256 == (
        "c3ad0b780dfbb3e5f8e1f7b7850c06fcb5c6d977e106f351b4602fcaadf210d2"
    )
    assert load_settings(CONFIG, task="isic2016").task_settings.output_size == 224
    assert load_settings(CONFIG, task="lsp").task_settings.output_size == 56


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("method", METHODS)
def test_all_three_tasks_and_four_methods_resolve(task: str, method: str) -> None:
    settings = load_settings(CONFIG, task=task, method=method, seed=2028)
    assert settings.task == task
    assert settings.method == method
    assert settings.seed == 2028
    assert settings.paths.run_dir == (
        settings.paths.output_root / task / method / "seed_2028"
    ).resolve()
    assert settings.paths.common_start_dir == (
        settings.paths.output_root / task / "common" / "seed_2028"
    ).resolve()
    assert settings.paths.common_start_checkpoint.name == "common_start.pt"
    assert settings.updates_backbone is (method != "noft")
    assert settings.run_epochs == 50
    expected_total = 50 if method == "noft" else 100
    assert settings.inherited_pipeline_epochs == expected_total


def test_paths_are_resolved_relative_to_repository_not_config_directory() -> None:
    settings = load_settings(CONFIG, task="isic2016", method="fa_pretrained")
    assert settings.repo_root == REPO_ROOT
    assert settings.paths.source_backbone.is_absolute()
    assert settings.paths.source_backbone == (
        REPO_ROOT
        / "experiments/qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone"
        / "runs/p11_imagenet1k_pretrain_bs96_90e/checkpoints/backbone.pt"
    ).resolve()
    assert settings.data_root == (REPO_ROOT / "data/ISIC2016").resolve()


def test_programmatic_overrides_are_limited_to_run_identity_and_limits() -> None:
    settings = load_settings(
        CONFIG,
        task="lsp",
        method="fa_random",
        seed=7,
        output_dir="experiments/custom_p12_smoke",
        limits={"max_train_batches": 2, "max_validation_batches": 1},
    )
    assert settings.output_dir == (REPO_ROOT / "experiments/custom_p12_smoke").resolve()
    assert settings.limits.max_train_batches == 2
    assert settings.limits.max_validation_batches == 1
    assert settings.limits.max_test_batches is None
    assert settings.training.adaptation_epochs == 50
    assert settings.optimizer.phase_learning_rate == pytest.approx(3.0e-3)


def test_output_root_override_isolates_smoke_common_start() -> None:
    settings = load_settings(
        CONFIG,
        task="caltech101",
        method="noft",
        seed=2026,
        output_root="experiments/p12_isolated_smoke",
    )
    expected = (REPO_ROOT / "experiments/p12_isolated_smoke").resolve()
    assert settings.paths.output_root == expected
    assert settings.output_dir == expected / "caltech101/noft/seed_2026"
    assert settings.paths.common_start_checkpoint == (
        expected / "caltech101/common/seed_2026/common_start.pt"
    )


def test_cli_overrides_return_the_same_resolved_dataclass() -> None:
    parser = add_settings_arguments(argparse.ArgumentParser())
    args = parser.parse_args(
        [
            "--config",
            str(CONFIG),
            "--task",
            "isic2016",
            "--method",
            "fa_pretrained",
            "--seed",
            "2027",
            "--max-train-batches",
            "3",
        ]
    )
    settings = load_settings_from_args(args)
    assert settings.task == "isic2016"
    assert settings.method == "fa_pretrained"
    assert settings.seed == 2027
    assert settings.limits.max_train_batches == 3


@pytest.mark.parametrize("task", ["imagenet", ""])
def test_unknown_task_is_rejected(task: str) -> None:
    with pytest.raises(ValueError, match="task must be one of"):
        load_settings(CONFIG, task=task)


def test_unknown_method_and_limit_are_rejected() -> None:
    with pytest.raises(ValueError, match="method must be one of"):
        load_settings(CONFIG, method="frozen_bp")
    with pytest.raises(ValueError, match="Unknown limit"):
        load_settings(CONFIG, limits={"max_epochs": 1})


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("training.head_only_epochs", 49, "head_only_epochs=50"),
        ("training.adaptation_epochs", 51, "adaptation_epochs=50"),
        ("model.num_stages", 7, "fixed P11 architecture"),
        ("model.mixer_width", 48, "fixed P11 architecture"),
        ("model.axis_schedule", ["token"] * 8, "fixed P11 architecture"),
        ("model.optical_gate_min", 0.49, "exactly 0.5"),
        ("optimizer.phase_learning_rate", 1.0e-4, "phase_learning_rate"),
        ("optimizer.phase_weight_decay", 1.0e-4, "zero weight decay"),
        ("tasks.isic2016.output_size", 256, "output_size must remain 224"),
        ("tasks.lsp.output_size", 224, "output_size must remain 56"),
        ("tasks.lsp.num_outputs", 17, "num_outputs=14"),
    ],
)
def test_scientific_protocol_changes_are_rejected(
    tmp_path: Path, key: str, value: object, message: str
) -> None:
    path = _mutated_config(tmp_path, key, value)
    task = "isic2016" if key.startswith("tasks.isic") else "lsp" if key.startswith("tasks.lsp") else None
    with pytest.raises(ValueError, match=message):
        load_settings(path, task=task)


@pytest.mark.parametrize("limits", [RunLimits(max_train_batches=0), {"max_test_samples": -1}])
def test_nonpositive_smoke_limits_are_rejected(limits: object) -> None:
    with pytest.raises(ValueError, match="must be positive or null"):
        load_settings(CONFIG, limits=limits)  # type: ignore[arg-type]


def test_resolved_settings_are_json_serializable() -> None:
    settings = load_settings(CONFIG, task="lsp", method="noft")
    payload = settings.to_dict()
    assert payload["task"] == "lsp"
    assert payload["paths"]["source_backbone"].endswith("backbone.pt")
    assert json.loads(settings.to_json())["training"]["head_only_epochs"] == 50
