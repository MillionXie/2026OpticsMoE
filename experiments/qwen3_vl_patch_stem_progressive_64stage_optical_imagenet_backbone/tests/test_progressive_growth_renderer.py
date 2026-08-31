from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from ..migration import P13_TRAINING_CHECKPOINT_FORMAT
from ..render_progressive_growth_config import (
    EXPECTED_GLOBAL_BATCH,
    canonical_json_sha256,
    inspect_parent_checkpoint,
    render_config,
    sha256_file,
    write_or_verify_config,
)


def save_parent(path: Path, *, depth: int, role: str = "best_full_depth") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    grandparent = 8 if depth == 16 else depth // 2
    depth_alpha = {
        "growth_parent_depth": grandparent,
        "carried_stage_count": grandparent,
        "new_stage_count": depth - grandparent,
        "minimum": 1.0,
        "maximum": 1.0,
        "mean": 1.0,
        "all_full_depth": True,
        "all_exact_bypass": False,
    }
    migration = {
        "format": "unit-migration",
        "source_depth": grandparent,
        "target_num_stages": depth,
        "target_architecture_signature": [13, 1, 2, depth],
    }
    initialization = {
        "mode": "unit-progressive",
        "source_depth": grandparent,
        "target_depth": depth,
        "migration": migration,
    }
    feedback_manifest = {
        "format": "unit-feedback",
        "method": "fa_source",
        "depth": depth,
        "connector_count": depth,
        "connections": [],
    }
    torch.save(
        {
            "format": P13_TRAINING_CHECKPOINT_FORMAT,
            "checkpoint_role": role,
            "model": {
                "p13_progressive_architecture_signature": torch.tensor(
                    [13, 1, 2, depth], dtype=torch.int64
                )
            },
            "model_config": {"num_stages": depth},
            "model_report": {
                "architecture": "p13_progressive_p11_token_channel",
                "num_stages": depth,
                "depth_alpha": depth_alpha,
                "migration_manifest": migration,
            },
            "stem_checkpoint_sha256": "stem-sha",
            "epoch": 20,
            "config_digest": f"config-depth-{depth}",
            "migration_manifest": migration,
            "initialization_manifest": initialization,
            "depth_alpha": depth_alpha,
            "feedback": {
                "method": "fa_source",
                "random_seed": None,
                "manifest": feedback_manifest,
                "manifest_sha256": canonical_json_sha256(feedback_manifest),
            },
        },
        path,
    )
    return path


def save_template(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "output_dir": "placeholder",
                "stem_checkpoint": "stem.pt",
                "model": {"num_stages": 16, "new_stage_ramp_epochs": 10},
                "initialization": {"mode": "must_be_replaced"},
                "feedback": {"method": "fa_source", "random_seed": None},
                "training": {
                    "epochs": 20,
                    "batch_size": 24,
                    "validation_batch_size": 48,
                    "gradient_accumulation_steps": 2,
                    "expected_world_size": 4,
                    "expected_effective_global_batch": 192,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_parent_guard_requires_immediate_best_full_depth(tmp_path: Path) -> None:
    parent = save_parent(
        tmp_path / "growth16" / "checkpoints" / "best_full_depth.pt",
        depth=16,
    )
    identity = inspect_parent_checkpoint(parent, target_depth=32)
    assert identity["source_depth"] == 16
    assert identity["target_depth"] == 32
    assert identity["sha256"] == sha256_file(parent)
    assert identity["feedback_method"] == "fa_source"

    with pytest.raises(RuntimeError, match="requires source depth 32"):
        inspect_parent_checkpoint(parent, target_depth=64)
    wrong_name = tmp_path / "growth16" / "checkpoints" / "last.pt"
    wrong_name.write_bytes(parent.read_bytes())
    with pytest.raises(RuntimeError, match="named best_full_depth"):
        inspect_parent_checkpoint(wrong_name, target_depth=32)

    wrong_role = save_parent(
        tmp_path / "wrong" / "best_full_depth.pt",
        depth=16,
        role="last",
    )
    with pytest.raises(RuntimeError, match="role must be best_full_depth"):
        inspect_parent_checkpoint(wrong_role, target_depth=32)


def test_rendered_config_pins_source_identity_and_global_batch(tmp_path: Path) -> None:
    parent = save_parent(
        tmp_path / "runs" / "growth16" / "checkpoints" / "best_full_depth.pt",
        depth=16,
    )
    template = save_template(tmp_path / "template.yaml")
    target_run = tmp_path / "runs" / "growth32"
    config, identity = render_config(
        template_config=template,
        parent_checkpoint=parent,
        target_depth=32,
        output_dir=target_run,
        repository=tmp_path,
    )

    assert config["model"]["num_stages"] == 32
    assert config["initialization"] == {
        "mode": "progressive_growth",
        "source_training_checkpoint": "runs/growth16/checkpoints/best_full_depth.pt",
        "expected_source_training_sha256": identity["sha256"],
        "expected_source_depth": 16,
        "expected_source_epoch": 20,
        "expected_source_config_digest": "config-depth-16",
        "expected_source_feedback_method": "fa_source",
        "expected_source_feedback_manifest_sha256": identity[
            "feedback_manifest_sha256"
        ],
    }
    training = config["training"]
    assert training["batch_size"] == 12
    assert training["gradient_accumulation_steps"] == 4
    assert (
        training["batch_size"]
        * training["gradient_accumulation_steps"]
        * training["expected_world_size"]
        == EXPECTED_GLOBAL_BATCH
    )
    guard = config["progressive_growth_guard"]
    assert guard["format"] == "p13-progressive-config-guard-v1"
    assert guard["parent_checkpoint_sha256"] == identity["sha256"]
    assert guard["source_depth"] == 16
    assert guard["target_depth"] == 32

    output = tmp_path / "configs" / "growth32.yaml"
    assert write_or_verify_config(output, config) == "rendered_new"
    assert write_or_verify_config(output, config) == "verified_existing"
    tampered = dict(config)
    tampered["output_dir"] = "another-run"
    with pytest.raises(RuntimeError, match="refusing overwrite"):
        write_or_verify_config(output, tampered)


def test_commands_fix_the_parent_chain_and_render_before_launch() -> None:
    experiment = Path(__file__).resolve().parents[1]
    helper = (experiment / "commands" / "_progressive_growth_common.sh").read_text(
        encoding="utf-8"
    )
    launcher = (experiment / "commands" / "13_launch_progressive_growth.sh").read_text(
        encoding="utf-8"
    )
    renderer = (
        experiment / "commands" / "12_render_or_verify_progressive_growth.sh"
    ).read_text(encoding="utf-8")

    assert '32)' in helper and 'PARENT_RUN_NAME="p13_growth16_' in helper
    assert '64)' in helper and 'PARENT_RUN_NAME="p13_growth32_' in helper
    assert '100)' in helper and 'PARENT_RUN_NAME="p13_growth64_' in helper
    assert "best_full_depth.pt" in helper
    assert "if [[ ! -f \"${PARENT_CHECKPOINT}\" ]]" in helper
    assert "render_or_verify_progressive_config" in launcher
    assert launcher.index("render_or_verify_progressive_config") < launcher.index(
        "nohup"
    )
    assert "TARGET_DEPTH=32, 64, or 100" in renderer
