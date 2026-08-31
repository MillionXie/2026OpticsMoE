from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.d2nn_cifar10_high_performance_optical_backbone.optics import (
    rms_normalize,
)

from ..migration import (
    P13_DEPTH_TRANSITIONS,
    P13_TRAINING_CHECKPOINT_FORMAT,
    migrate_strict_p11_training_checkpoint,
    migrate_strict_progressive_checkpoint,
    progressive_pair_mapping,
    progressive_stage_mapping,
)
from ..model import (
    QwenStemProgressiveOpticalImageNetBackbone,
    anchor_stage_indices,
)
from .test_progressive_model import common_config, fake_stem, p11_export


def save_p11_best(
    path: Path,
    source,
    *,
    config_digest: str = "unit-test-config",
) -> Path:
    torch.save(
        {
            "model": source.state_dict(),
            "optimizer": {},
            "scheduler": {},
            "scaler": {},
            "epoch": 88,
            "best_validation_top1": 0.51348,
            "history": [],
            "config_digest": config_digest,
        },
        path,
    )
    return path


def save_p13_training_checkpoint(path: Path, model, *, epoch: int = 20) -> Path:
    torch.save(
        {
            "format": P13_TRAINING_CHECKPOINT_FORMAT,
            "checkpoint_role": "best_full_depth",
            "model": model.state_dict(),
            "model_config": model.config,
            "model_report": model.parameter_report(),
            "stem_checkpoint_sha256": model.stem.checkpoint_sha256,
            "epoch": epoch,
            "config_digest": f"unit-depth-{model.num_stages}",
            "migration_manifest": {"format": "unit-parent-migration"},
            "depth_alpha": model.depth_alpha_report(),
        },
        path,
    )
    return path


def synthetic_field(seed: int = 917) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return rms_normalize(
        torch.rand(1, 3, 224, 224, generator=generator),
        1.0e-5,
    )


def test_growth_provenance_parameter_groups_and_alpha_are_disjoint(
    tmp_path: Path,
) -> None:
    config = common_config()
    config.update({"num_stages": 32, "new_stage_alpha_init": 0.0})
    stem = fake_stem(tmp_path / "stem.pt")
    model = QwenStemProgressiveOpticalImageNetBackbone(
        stem,
        config,
    )
    stage_mapping = progressive_stage_mapping(16, 32)
    growth = [-1] * 32
    for source_index, target_index in enumerate(stage_mapping):
        growth[target_index] = source_index
    model.set_growth_parent_stage_indices(growth, new_stage_alpha=0.0)

    assert len(model.mixer_anchor_slots()) == 8
    assert len(model.carried_slots()) == 16
    assert len(model.new_slots()) == 16
    assert all(slot.is_p11_mixer_anchor for slot in model.mixer_anchor_slots())
    assert all(slot.is_carried_from_parent for slot in model.carried_slots())
    assert all(slot.is_newly_inserted for slot in model.new_slots())
    assert all(slot.alpha_value == 1.0 for slot in model.carried_slots())
    assert all(slot.alpha_value == 0.0 for slot in model.new_slots())

    groups = [
        list(model.carried_phase_parameters()),
        list(model.new_phase_parameters()),
        list(model.carried_electronic_parameters()),
        list(model.new_electronic_parameters()),
        list(model.head_parameters()),
    ]
    identifiers = [{id(parameter) for parameter in group} for group in groups]
    for index, values in enumerate(identifiers):
        assert values
        assert all(not values.intersection(other) for other in identifiers[index + 1 :])
    grouped = set().union(*identifiers)
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert grouped == trainable

    model.apply_depth_ramp(3)
    assert all(slot.alpha_value == 1.0 for slot in model.carried_slots())
    assert all(0.0 < slot.alpha_value < 1.0 for slot in model.new_slots())

    # Growth provenance is controller state that must survive an ordinary
    # state-dict save/load, not a transient Python annotation reconstructed
    # from the immutable P11 mixer anchors.
    reloaded = QwenStemProgressiveOpticalImageNetBackbone(stem, config)
    reloaded.load_state_dict(model.state_dict(), strict=True)
    assert torch.equal(
        reloaded.growth_parent_stage_indices,
        model.growth_parent_stage_indices,
    )
    assert [slot.stage_index for slot in reloaded.carried_slots()] == [
        slot.stage_index for slot in model.carried_slots()
    ]
    assert [slot.stage_index for slot in reloaded.new_slots()] == [
        slot.stage_index for slot in model.new_slots()
    ]
    assert [slot.alpha_value for slot in reloaded.slots] == [
        slot.alpha_value for slot in model.slots
    ]

    snapshot = model.phase_snapshot()
    assert model.phase_motion(snapshot)["mean_absolute_rad"] == 0.0
    with torch.no_grad():
        model.new_slots()[0].stage.raw_phase[0, 0, 0].add_(0.2)
    assert model.phase_motion(snapshot)["mean_absolute_rad"] > 0.0


def test_p11_best_and_backbone_cross_check_preserves_features_and_logits(
    tmp_path: Path,
) -> None:
    stem = fake_stem(tmp_path / "stem.pt")
    backbone, source = p11_export(tmp_path / "p11_backbone.pt", stem)
    best = save_p11_best(tmp_path / "p11_best.pt", source)
    config = common_config()
    config.update({"num_stages": 16, "new_stage_alpha_init": 0.0})
    target = QwenStemProgressiveOpticalImageNetBackbone(stem, config)
    manifest = migrate_strict_p11_training_checkpoint(target, backbone, best)
    source.eval()
    target.eval()

    amplitude = synthetic_field()
    with torch.no_grad():
        expected = amplitude
        for stage in source.stages:
            expected = stage(expected)
        actual, _ = target.forward_field(amplitude)
        expected_logits = source.readout(expected, source.stem.token_count)
        actual_logits = target.readout(actual, target.stem.token_count)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_logits, expected_logits, rtol=0.0, atol=0.0)
    assert manifest["source_imagenet_head_migrated"] is True
    assert manifest["source_best_epoch"] == 88
    assert manifest["source_best_validation_top1"] == pytest.approx(0.51348)
    assert target.growth_parent_depth == 8
    assert len(target.carried_slots()) == 8
    assert len(target.new_slots()) == 8
    assert torch.equal(target.feedback_source_snapshot(), target.phase_snapshot())
    assert target.feedback_manifest()["method"] == "bp_current"


def test_p11_best_nonhead_mismatch_is_rejected_before_migration(
    tmp_path: Path,
) -> None:
    stem = fake_stem(tmp_path / "stem.pt")
    backbone, source = p11_export(tmp_path / "p11_backbone.pt", stem)
    state = {name: value.clone() for name, value in source.state_dict().items()}
    state["adapter.projection.weight"][0, 0].add_(1.0)
    best = tmp_path / "corrupt_best.pt"
    torch.save(
        {
            "model": state,
            "optimizer": {},
            "scheduler": {},
            "scaler": {},
            "epoch": 88,
            "best_validation_top1": 0.51348,
            "history": [],
            "config_digest": "unit-test-config",
        },
        best,
    )
    config = common_config()
    config.update({"num_stages": 16, "new_stage_alpha_init": 0.0})
    target = QwenStemProgressiveOpticalImageNetBackbone(stem, config)
    before = target.phase_snapshot()
    with pytest.raises(RuntimeError, match="non-head tensors differ"):
        migrate_strict_p11_training_checkpoint(target, backbone, best)
    torch.testing.assert_close(target.phase_snapshot(), before, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(("source_depth", "target_depth"), P13_DEPTH_TRANSITIONS)
def test_progressive_pair_mapping_is_monotone_and_pins_mixer_anchors(
    source_depth: int,
    target_depth: int,
) -> None:
    pairs = progressive_pair_mapping(source_depth, target_depth)
    assert len(pairs) == source_depth // 2
    assert list(pairs) == sorted(set(pairs))
    source_anchors = [value // 2 for value in anchor_stage_indices(source_depth)[::2]]
    target_anchors = [value // 2 for value in anchor_stage_indices(target_depth)[::2]]
    assert [pairs[index] for index in source_anchors] == target_anchors


@pytest.mark.parametrize(("source_depth", "target_depth"), P13_DEPTH_TRANSITIONS)
def test_strict_progressive_checkpoint_migrates_every_parent_stage(
    tmp_path: Path,
    source_depth: int,
    target_depth: int,
) -> None:
    stem = fake_stem(tmp_path / "stem.pt")
    source_config = common_config()
    source_config.update(
        {
            "num_stages": source_depth,
            "new_stage_alpha_init": 0.0,
            "new_stage_ramp_epochs": 2,
        }
    )
    source = QwenStemProgressiveOpticalImageNetBackbone(stem, source_config)
    source.apply_depth_ramp(2)
    source.eval()
    checkpoint = save_p13_training_checkpoint(
        tmp_path / f"source_{source_depth}.pt",
        source,
    )
    target_config = dict(source_config)
    target_config["num_stages"] = target_depth
    target = QwenStemProgressiveOpticalImageNetBackbone(stem, target_config)
    manifest = migrate_strict_progressive_checkpoint(target, checkpoint)

    assert manifest["source_depth"] == source_depth
    assert manifest["target_num_stages"] == target_depth
    assert len(target.carried_slots()) == source_depth
    assert len(target.new_slots()) == target_depth - source_depth
    assert [
        slot.growth_parent_stage_index for slot in target.carried_slots()
    ] == list(range(source_depth))
    assert all(slot.alpha_value == 1.0 for slot in target.carried_slots())
    assert all(slot.alpha_value == 0.0 for slot in target.new_slots())
    assert torch.equal(target.feedback_source_snapshot(), target.phase_snapshot())
    assert target.feedback_manifest()["source"]["connector_count"] == target_depth

    target.eval()
    amplitude = synthetic_field(seed=1193)
    with torch.no_grad():
        expected, _ = source.forward_field(amplitude)
        actual, _ = target.forward_field(amplitude)
        expected_logits = source.readout(expected, source.stem.token_count)
        actual_logits = target.readout(actual, target.stem.token_count)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        actual_logits,
        expected_logits,
        rtol=0.0,
        atol=0.0,
    )


def test_progressive_checkpoint_rejects_parent_before_alpha_one(
    tmp_path: Path,
) -> None:
    stem = fake_stem(tmp_path / "stem.pt")
    source_config = common_config()
    source_config.update({"num_stages": 16, "new_stage_alpha_init": 0.1})
    source = QwenStemProgressiveOpticalImageNetBackbone(stem, source_config)
    checkpoint = save_p13_training_checkpoint(tmp_path / "partial.pt", source)
    target_config = dict(source_config)
    target_config.update({"num_stages": 32, "new_stage_alpha_init": 0.0})
    target = QwenStemProgressiveOpticalImageNetBackbone(stem, target_config)
    with pytest.raises(RuntimeError, match="not at alpha-one full depth"):
        migrate_strict_progressive_checkpoint(target, checkpoint)


def test_progressive_checkpoint_rejects_nonselected_checkpoint_role(
    tmp_path: Path,
) -> None:
    stem = fake_stem(tmp_path / "stem.pt")
    source_config = common_config()
    source_config.update({"num_stages": 16, "new_stage_alpha_init": 0.0})
    source = QwenStemProgressiveOpticalImageNetBackbone(stem, source_config)
    source.apply_depth_ramp(source.new_stage_ramp_epochs)
    checkpoint = save_p13_training_checkpoint(tmp_path / "source.pt", source)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["checkpoint_role"] = "last"
    torch.save(payload, checkpoint)

    target_config = dict(source_config)
    target_config["num_stages"] = 32
    target = QwenStemProgressiveOpticalImageNetBackbone(stem, target_config)
    with pytest.raises(RuntimeError, match="best_full_depth"):
        migrate_strict_progressive_checkpoint(target, checkpoint)
