import copy
import json

import pytest

from LightGenV2.tasks.t06_video_quality_assessment import prepare_vlm_assets as assets


def profile():
    return json.loads((assets.shared.TASK / "configs/baselines/internvl35_temporal_4f.json").read_text())


def test_official_fixed_revision_and_training_budget():
    p = profile()
    assets.validate_profile(p)
    assert p["epochs"] == 50 and p["feature_batch_size"] == 2
    assert p["hidden_width"] * 5 == p["quality_output_trainable_parameters"] == 10240
    assert p["backbone_trainable_parameters"] == 0
    assert p["frames"] == 4 and p["train_samples"] == 2250 and p["test_samples"] == 558
    assert len(p["official_assets"]) == 13


def test_ovis_candidate_is_same_budget_and_metadata_precedes_weight_shards():
    p = json.loads((assets.shared.TASK / "configs/baselines/ovis25_temporal_4f.json").read_text())
    assets.validate_profile(p)
    assert p["frames"] == 4 and p["epochs"] == 50
    assert p["inference_batches"] == [1] and p["thinking"] is False
    assert p["hidden_width"]*5 == p["quality_output_trainable_parameters"] == 10240
    ordered = sorted(p["official_assets"], key=lambda n: n.endswith(".safetensors"))
    assert ordered[-2:] == ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]


@pytest.mark.parametrize("kind", ["revision", "path", "size", "blob"])
def test_unsafe_or_unpinned_assets_rejected(kind):
    p = copy.deepcopy(profile())
    if kind == "revision":
        p["model_revision"] = "main"
    elif kind == "path":
        p["official_assets"]["../bad"] = {"git_blob_sha1": "0" * 40}
    elif kind == "size":
        p["official_assets"]["model.safetensors"]["size"] = 0
    else:
        p["official_assets"]["config.json"]["git_blob_sha1"] = "bad"
    with pytest.raises(ValueError):
        assets.validate_profile(p)
