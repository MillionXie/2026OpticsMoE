from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from experiments.qwen3_vl_patch_stem_8stage_separable_optical_imagenet22k_backbone.dataset import (
    DatasetContractError,
)
from experiments.qwen3_vl_patch_stem_8stage_separable_optical_imagenet22k_backbone.train import (
    IMPLEMENTATION_FILES,
    REPOSITORY_ROOT,
    _taxonomy_identity,
    _validate_index_against_config,
    accumulation_window_size,
    amp_optimizer_step_succeeded,
    implementation_manifest,
    preflight,
    soft_target_cross_entropy,
    training_rank_seed,
)


PROJECT = Path(__file__).resolve().parents[1]


def test_soft_target_ce_matches_hard_ce() -> None:
    logits = torch.tensor([[2.0, -1.0, 0.5], [-0.2, 0.1, 1.7]])
    labels = torch.tensor([0, 2])
    targets = torch.nn.functional.one_hot(labels, num_classes=3).float()
    assert torch.allclose(
        soft_target_cross_entropy(logits, targets),
        torch.nn.functional.cross_entropy(logits, labels),
    )


def test_training_rng_is_rank_specific_after_shared_model_seed() -> None:
    seeds = [training_rank_seed(2026, rank) for rank in range(5)]
    assert seeds[0] == 2026
    assert len(set(seeds)) == 5


def test_tail_accumulation_uses_true_window_divisor() -> None:
    assert [accumulation_window_size(index, 10, 4) for index in range(1, 11)] == [
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        2,
        2,
    ]


def test_amp_overflow_does_not_count_as_optimizer_step() -> None:
    assert amp_optimizer_step_succeeded(256.0, 256.0)
    assert amp_optimizer_step_succeeded(256.0, 512.0)
    assert not amp_optimizer_step_succeeded(256.0, 128.0)


def test_implementation_manifest_includes_module_path_mapping() -> None:
    assert "experiments/__init__.py" in IMPLEMENTATION_FILES
    assert all((REPOSITORY_ROOT / relative).is_file() for relative in IMPLEMENTATION_FILES)
    value = implementation_manifest()
    assert value["aggregate_sha256"]
    assert {row["path"] for row in value["files"]} == set(IMPLEMENTATION_FILES)


def _index_manifest(source_root: Path, **overrides):
    value = {
        "source_root": str(source_root),
        "variant_id": "miil-imagenet21k-p-fall11",
        "release_id": "fall11",
        "split_id": "train",
        "num_classes": 11_221,
        "num_samples": 11_797_632,
        "class_list_sha256": "a" * 64,
        "class_to_idx_sha256": "b" * 64,
    }
    value.update(overrides)
    return value


def _index_config():
    return {
        "variant_id": "miil-imagenet21k-p-fall11",
        "release_id": "fall11",
        "num_classes": 11_221,
        "train_split_id": "train",
        "validation_split_id": "validation",
        "expected_train_samples": 11_797_632,
        "expected_validation_samples": 561_052,
    }


def test_index_contract_requires_live_source_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="source_root"):
        _validate_index_against_config(
            _index_manifest(tmp_path / "missing"),
            _index_config(),
            role="train",
        )


def test_index_contract_locks_split_id(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(DatasetContractError, match="split_id"):
        _validate_index_against_config(
            _index_manifest(source, split_id="validation"),
            _index_config(),
            role="train",
        )


@pytest.mark.parametrize("key", ["class_list_sha256", "class_to_idx_sha256"])
def test_train_validation_taxonomy_hashes_must_match(tmp_path: Path, key: str) -> None:
    train = _index_manifest(tmp_path)
    validation = _index_manifest(
        tmp_path,
        split_id="validation",
        num_samples=561_052,
    )
    validation[key] = "c" * 64
    with pytest.raises(DatasetContractError, match="taxonomy mismatch"):
        _taxonomy_identity([train, validation])


def test_common_taxonomy_digest_covers_both_hashes(tmp_path: Path) -> None:
    train = _index_manifest(tmp_path)
    validation = _index_manifest(tmp_path, split_id="validation", num_samples=561_052)
    identity = _taxonomy_identity([train, validation])
    assert identity["class_list_sha256"] == "a" * 64
    assert identity["class_to_idx_sha256"] == "b" * 64
    assert len(identity["taxonomy_digest"]) == 64


def test_configs_use_soft_ce_and_smoke_is_non_publishable() -> None:
    smoke = yaml.safe_load(
        (PROJECT / "configs" / "plumbing_smoke_imagenet1k_21841_gpu5.yaml").read_text(
            encoding="utf-8"
        )
    )
    full = yaml.safe_load(
        (PROJECT / "configs" / "imagenet22k_fall11_21841_90e.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert smoke["dataset"]["publishable_result"] is False
    assert smoke["training"]["max_train_batches"] == 100
    assert smoke["model"]["num_classes"] == 21_841
    assert full["loss"]["mode"] == "soft_target_cross_entropy"
    assert full["evaluation"]["enabled"] is False
    assert full["training"]["epochs"] == 90
    common = (PROJECT / "commands" / "_common.sh").read_text(encoding="utf-8")
    assert "--verify-large-index-files" in common


def test_formal_preflight_fails_before_output_when_index_absent(tmp_path: Path) -> None:
    output = tmp_path / "must_not_be_created"
    config = {
        "output_dir": str(output),
        "stem_checkpoint": str(tmp_path / "stem.pt"),
        "dataset": {
            "mode": "indexed_class_folder",
            "train_index": str(tmp_path / "missing-index"),
            "validation_index": None,
            "variant_id": "imagenet-fall11-full",
            "release_id": "fall11",
            "num_classes": 21_841,
            "expected_train_samples": 14_197_122,
        },
        "evaluation": {"enabled": False},
        "model": {"num_classes": 21_841},
        "loss": {"mode": "soft_target_cross_entropy"},
        "initialization": {
            "backbone_checkpoint": str(tmp_path / "backbone.pt"),
            "expected_backbone_sha256": "0" * 64,
            "expected_stem_sha256": "1" * 64,
        },
        "training": {},
    }
    with pytest.raises(FileNotFoundError, match="Formal large-data index is absent"):
        preflight(config)
    assert not output.exists()
