from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from experiments.qwen3_vl_patch_stem_8stage_optical_imagenet_backbone.train import (
    load_config,
)

from .. import clean_recovery
from ..large_scale_continue import classification_loss


def _source_payload() -> dict:
    return {
        "format": clean_recovery.EXPECTED_SOURCE_FORMAT,
        "checkpoint_role": clean_recovery.EXPECTED_SOURCE_ROLE,
        "epoch": clean_recovery.EXPECTED_SOURCE_EPOCH,
        "config_digest": clean_recovery.EXPECTED_SOURCE_CONFIG_DIGEST,
        "model": {
            "weight": torch.tensor([1.0]),
            "frozen": torch.tensor([3.0]),
        },
        "ema": {"shadow": {"weight": torch.tensor([2.0])}},
    }


def test_source_payload_identity_is_strict() -> None:
    clean_recovery.validate_source_payload(_source_payload())
    for key, value in (
        ("format", "wrong"),
        ("checkpoint_role", "best_raw"),
        ("epoch", 4),
        ("config_digest", "wrong"),
    ):
        payload = _source_payload()
        payload[key] = value
        with pytest.raises(RuntimeError, match=key):
            clean_recovery.validate_source_payload(payload)


def test_source_state_variant_selects_raw_or_ema_without_mutating_source() -> None:
    payload = _source_payload()
    original = copy.deepcopy(payload)
    raw = clean_recovery.select_source_state(payload, "raw")
    ema = clean_recovery.select_source_state(payload, "ema")
    torch.testing.assert_close(raw["weight"], torch.tensor([1.0]))
    torch.testing.assert_close(ema["weight"], torch.tensor([2.0]))
    torch.testing.assert_close(ema["frozen"], torch.tensor([3.0]))
    torch.testing.assert_close(payload["model"]["weight"], original["model"]["weight"])
    with pytest.raises(ValueError, match="raw or ema"):
        clean_recovery.select_source_state(payload, "teacher")


def test_source_initializer_strict_loads_raw_and_starts_new_optimizer(
    tmp_path, monkeypatch
) -> None:
    source_path = tmp_path / "last.pt"
    payload = _source_payload()
    payload.update(
        {
            "model_config": {},
            "stem_checkpoint_sha256": "stem-sha",
            "best_raw_top1": 0.52,
            "best_ema_top1": 0.51,
            "global_optimizer_step": 123,
            "initial_phases_sha256": "source-phase-sha",
        }
    )
    torch.save(payload, source_path)
    monkeypatch.setattr(
        clean_recovery, "sha256_file", lambda _: clean_recovery.EXPECTED_SOURCE_SHA256
    )

    class FakeStem:
        checkpoint_sha256 = "stem-sha"

    class FakeModel:
        stem = FakeStem()
        loaded = None

        def load_state_dict(self, state, strict=False):
            self.loaded = dict(state)
            assert strict is True

    model = FakeModel()
    initialization = clean_recovery.initialize_from_completed_proxy(
        model,  # type: ignore[arg-type]
        {
            "initialization": {
                "source_checkpoint": str(source_path),
                "source_state_variant": "raw",
            },
            "model": {},
        },
    )
    torch.testing.assert_close(model.loaded["weight"], torch.tensor([1.0]))
    assert initialization["source_checkpoint_epoch"] == 5
    assert initialization["source_state_variant"] == "raw"
    assert initialization["source_optimizer_reused"] is False
    assert initialization["clean_optimizer_initialized_fresh"] is True


def test_registered_config_is_clean_ce_and_single_gpu() -> None:
    config = load_config(
        "FixedFeedbackSFT/projects/"
        "qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone/"
        "configs/clean_recovery_5e_raw_gpu5_gb96.yaml"
    )

    class FakeContext:
        world_size = 1

    clean_recovery.validate_clean_config(config, FakeContext())  # type: ignore[arg-type]
    assert config["initialization"]["source_state_variant"] == "raw"
    assert config["optimizer"]["phase_learning_rate"] == 8.0e-4
    assert config["loss"]["batch_mix_probability"] == 0.0
    assert config["model"]["stage_drop_path_rate"] == 0.0

    logits = torch.randn(5, 11)
    labels = torch.tensor([0, 1, 2, 3, 4])
    actual = classification_loss(logits, labels, labels, 1.0, config)
    expected = F.cross_entropy(logits, labels, label_smoothing=0.0)
    assert math.isclose(float(actual), float(expected), rel_tol=1.0e-7)


def test_registered_source_identity_constants_cannot_be_overridden_by_yaml() -> None:
    config = load_config(
        "FixedFeedbackSFT/projects/"
        "qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone/"
        "configs/clean_recovery_5e_raw_gpu5_gb96.yaml"
    )
    assert "expected_source_sha256" not in config["initialization"]
    assert clean_recovery.EXPECTED_SOURCE_SHA256 == (
        "34175ba9e764b7eef5bd59b1e1d1dd7f602281d02bd709ebf12ec55c0338f681"
    )
    assert clean_recovery.EXPECTED_SOURCE_CONFIG_DIGEST == (
        "8bea33ea2f6cccf25b499ff5949eb5462d232b11694e29bba9c1b8dccb8ba202"
    )


def test_clean_recovery_implementation_manifest_paths_exist() -> None:
    repository_root = Path(__file__).resolve().parents[4]
    missing = [
        relative
        for relative in clean_recovery.IMPLEMENTATION_FILES
        if not (repository_root / relative).is_file()
    ]
    assert missing == []
