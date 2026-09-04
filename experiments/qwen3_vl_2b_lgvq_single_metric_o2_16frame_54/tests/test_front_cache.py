from __future__ import annotations

import csv
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from ..cache_qwen_front import (
    FRAME_FRACTIONS,
    _language_embedding_fingerprint,
    _tensor_stream_sha256,
    _vision_front_fingerprint,
    pool_qwen_front_tokens,
    quality_tokens_from_images,
    render_prompt,
)
from ..data import (
    QWEN_COMPONENT_FINGERPRINT_CONTRACT,
    QWEN_FRONT_PAIR_CONTRACT,
    QWEN_SOURCE_IDENTITY_CONTRACT,
    _validate_cache_front_identity,
    LGVQSingleMetricDataset,
    load_single_metric_cache,
)
from ..settings import (
    FEATURE_CONTRACT,
    LANGUAGE_CONTRACT,
    QUALITY_CONTRACT,
    TARGET_PROMPTS,
    load_settings,
)


ROOT = Path(__file__).resolve().parents[1]


def _record(**payload):
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {**payload, "sha256": hashlib.sha256(encoded).hexdigest()}


def _fake_front_identity(tag: str = "same") -> dict[str, dict]:
    source = _record(
        contract=QWEN_SOURCE_IDENTITY_CONTRACT,
        requested_model="Qwen/Qwen3-VL-2B-Instruct",
        resolved_model_path="/models/qwen3-vl-2b",
        checkpoint_revision=tag,
        model_class="Qwen3VLForConditionalGeneration",
        transformers_version="test",
        model_config_sha256=(tag.encode().hex() + "0" * 64)[:64],
        checkpoint_artifact_manifest_sha256=("a" + "0" * 63),
        checkpoint_artifacts={"files": []},
    )
    vision = _record(
        contract=QWEN_COMPONENT_FINGERPRINT_CONTRACT,
        component="vision_patch_embed_and_position",
        source_identity_sha256=source["sha256"],
        vision_config={"hidden_size": 1024},
        position_interpolation_code_sha256="b" * 64,
        tensors=[{"name": "patch_embed.weight", "sha256": "c" * 64}],
    )
    language = _record(
        contract=QWEN_COMPONENT_FINGERPRINT_CONTRACT,
        component="language_embed_tokens",
        source_identity_sha256=source["sha256"],
        language_config={"hidden_size": 2048},
        tensors=[{"name": "weight", "sha256": "d" * 64}],
    )
    pair = _record(
        contract=QWEN_FRONT_PAIR_CONTRACT,
        source_identity_sha256=source["sha256"],
        vision_front_sha256=vision["sha256"],
        language_embedding_sha256=language["sha256"],
    )
    return {"source": source, "vision": vision, "language": language, "pair": pair}


def _attach_vision_identity(payload: dict, identity: dict[str, dict]) -> None:
    payload.update(
        {
            "qwen_front_contract": {"vision_hidden_size": 1024, "language_hidden_size": 2048},
            "qwen_source_identity": identity["source"],
            "qwen_vision_front_fingerprint": identity["vision"],
            "qwen_front_pair_identity": identity["pair"],
        }
    )


def _attach_language_identity(payload: dict, identity: dict[str, dict]) -> None:
    payload.update(
        {
            "qwen_front_contract": {"vision_hidden_size": 1024, "language_hidden_size": 2048},
            "qwen_source_identity": identity["source"],
            "qwen_language_embedding_fingerprint": identity["language"],
            "qwen_front_pair_identity": identity["pair"],
        }
    )


def test_release_configs_are_independent_single_targets() -> None:
    spatial = load_settings(ROOT / "configs/release/spatial.yaml")
    temporal = load_settings(ROOT / "configs/release/temporal.yaml")
    assert spatial.target_name == "spatial"
    assert temporal.target_name == "temporal"
    assert spatial.prompt == TARGET_PROMPTS["spatial"]
    assert temporal.prompt == TARGET_PROMPTS["temporal"]
    assert spatial.vision_cache_path != temporal.vision_cache_path
    assert spatial.language_cache_path != temporal.language_cache_path
    assert spatial.frame_count == 4
    assert temporal.frame_count == 16
    assert spatial.token_count == temporal.token_count == 49
    assert spatial.geometry.lane_origins == tuple(
        (top, left)
        for top in (0, 246)
        for left in (0, 246)
    )
    assert temporal.geometry.lane_origins == tuple(
        (top, left)
        for top in (2, 122, 242, 362)
        for left in (2, 122, 242, 362)
    )
    assert spatial.geometry.parallel_expert_size == 109
    assert temporal.geometry.parallel_expert_size == 54
    assert spatial.geometry.parallel_expert_pitch == 123
    assert temporal.geometry.parallel_expert_pitch == 60


def test_temporal36_release_config_keeps_the_same_active_field() -> None:
    temporal = load_settings(ROOT / "configs/release/temporal36.yaml")
    assert temporal.target_name == "temporal"
    assert temporal.frame_count == 36
    assert temporal.geometry.canvas_size == 518
    assert temporal.geometry.active_size == 478
    assert temporal.geometry.lane_grid == 6
    assert temporal.geometry.parallel_expert_size == 37
    assert len(temporal.geometry.lane_origins) == 36


def test_sixteen_frame_fractions_cover_central_span() -> None:
    assert len(FRAME_FRACTIONS) == 16
    assert FRAME_FRACTIONS[0] == pytest.approx(0.10)
    assert FRAME_FRACTIONS[-1] == pytest.approx(0.90)
    differences = np.diff(FRAME_FRACTIONS)
    assert np.allclose(differences, differences[0])


def test_two_stage_qwen_pooling_is_exact() -> None:
    hidden = torch.arange(2 * 784, dtype=torch.float32).view(-1, 1).expand(-1, 1024)
    pooled = pool_qwen_front_tokens(hidden, image_count=2)
    assert pooled.shape == (2, 49, 1024)
    first_196 = hidden[:784].reshape(196, 4, 1024).mean(1).reshape(14, 14, 1024)
    expected_first = first_196[:2, :2].mean((0, 1))
    torch.testing.assert_close(pooled[0, 0], expected_first)


def test_qwen_native_14x14_grouping_is_exact() -> None:
    hidden = torch.arange(784, dtype=torch.float32).view(-1, 1).expand(-1, 1024)
    pooled = pool_qwen_front_tokens(hidden, image_count=1, output_grid=14)
    assert pooled.shape == (1, 196, 1024)
    expected = hidden.reshape(196, 4, 1024).mean(1)
    torch.testing.assert_close(pooled[0], expected)


def test_quality_bank_is_14_channel_7x7_and_temporal() -> None:
    images = []
    for index in range(16):
        value = np.full((28, 28, 3), index * 10, dtype=np.uint8)
        images.append(Image.fromarray(value))
    tokens = quality_tokens_from_images(images, video_count=1)
    assert tokens.shape == (1, 16, 49, 14)
    assert tokens.dtype == torch.float16
    assert bool(torch.isfinite(tokens).all())
    # Channel 10 is the previous-frame luminance absolute difference.
    assert torch.count_nonzero(tokens[0, 0, :, 10]) == 0
    assert float(tokens[0, 1, :, 10].mean()) > 0.0


def test_quality_bank_supports_spatial_14x14() -> None:
    images = [Image.fromarray(np.full((28, 28, 3), 127, dtype=np.uint8)) for _ in range(4)]
    tokens = quality_tokens_from_images(
        images, video_count=1, frame_count=4, output_grid=14
    )
    assert tokens.shape == (1, 4, 196, 14)


def test_chat_template_is_target_bound() -> None:
    class Tokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            assert not tokenize and add_generation_prompt
            return messages[0]["content"][0]["text"] + "<assistant>"

    rendered = render_prompt(
        Tokenizer(),
        target_name="spatial",
        prompt=TARGET_PROMPTS["spatial"],
    )
    assert "spatial quality" in rendered
    with pytest.raises(ValueError, match="does not match"):
        render_prompt(
            Tokenizer(),
            target_name="spatial",
            prompt=TARGET_PROMPTS["temporal"],
        )


def test_tensor_fingerprint_is_chunk_invariant_and_content_sensitive() -> None:
    value = torch.arange(15 * 7, dtype=torch.float32).reshape(15, 7)
    assert _tensor_stream_sha256(value, chunk_bytes=28) == _tensor_stream_sha256(
        value, chunk_bytes=4096
    )
    changed = value.clone()
    changed[8, 3] += 1
    assert _tensor_stream_sha256(changed, chunk_bytes=28) != _tensor_stream_sha256(
        value, chunk_bytes=28
    )


def test_component_fingerprints_cover_patch_position_and_embed_tokens() -> None:
    class Config:
        def __init__(self, hidden_size: int):
            self.hidden_size = hidden_size

        def to_dict(self):
            return {"hidden_size": self.hidden_size, "spatial_merge_size": 2}

    class Visual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = torch.nn.Linear(3, 4, bias=False)
            self.pos_embed = torch.nn.Embedding(9, 4)
            self.config = Config(4)

        def fast_pos_embed_interpolate(self, grid_thw):
            return self.pos_embed.weight[: int(grid_thw.shape[0])]

    visual = Visual()
    source = "1" * 64
    first = _vision_front_fingerprint(visual, source_identity_sha256=source)
    assert {item["name"] for item in first["tensors"]} == {
        "patch_embed.weight",
        "pos_embed.weight",
    }
    with torch.no_grad():
        visual.pos_embed.weight[0, 0].add_(0.25)
    second = _vision_front_fingerprint(visual, source_identity_sha256=source)
    assert first["sha256"] != second["sha256"]

    embedding = torch.nn.Embedding(11, 6)
    language_first = _language_embedding_fingerprint(
        embedding,
        source_identity_sha256=source,
        language_config=Config(6),
    )
    with torch.no_grad():
        embedding.weight[5, 2].add_(1.0)
    language_second = _language_embedding_fingerprint(
        embedding,
        source_identity_sha256=source,
        language_config=Config(6),
    )
    assert language_first["sha256"] != language_second["sha256"]


def test_front_identity_rejects_cross_checkpoint_cache_mix() -> None:
    first = _fake_front_identity("revision-a")
    second = _fake_front_identity("revision-b")
    vision: dict = {}
    language: dict = {}
    _attach_vision_identity(vision, first)
    _attach_language_identity(language, second)
    with pytest.raises(RuntimeError, match="same Qwen checkpoint/revision"):
        _validate_cache_front_identity(vision, language)


def test_front_identity_rejects_modified_audit_record() -> None:
    identity = _fake_front_identity()
    vision: dict = {}
    language: dict = {}
    _attach_vision_identity(vision, identity)
    _attach_language_identity(language, identity)
    vision["qwen_source_identity"] = copy.deepcopy(vision["qwen_source_identity"])
    vision["qwen_source_identity"]["checkpoint_revision"] = "tampered"
    with pytest.raises(RuntimeError, match="content digest is invalid"):
        _validate_cache_front_identity(vision, language)


def test_cache_loader_returns_one_scalar_and_quality_tokens(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sample_id", "video_path", "split", "spatial", "temporal"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "a",
                "video_path": "a.mp4",
                "split": "train",
                "spatial": 71,
                "temporal": 31,
            }
        )
        writer.writerow(
            {
                "sample_id": "b",
                "video_path": "b.mp4",
                "split": "test",
                "spatial": 52,
                "temporal": 92,
            }
        )
    vision = tmp_path / "vision.pt"
    identity = _fake_front_identity()
    vision_payload = {
        "schema_version": 1,
        "feature_contract": FEATURE_CONTRACT,
        "quality_contract": QUALITY_CONTRACT,
        "frame_count": 16,
        "vision_tokens": torch.zeros(2, 16, 49, 1024, dtype=torch.float16),
        "quality_tokens": torch.zeros(2, 16, 49, 14, dtype=torch.float16),
        "sample_ids": ["a", "b"],
        "video_paths": ["a.mp4", "b.mp4"],
        "splits": ["train", "test"],
    }
    _attach_vision_identity(vision_payload, identity)
    torch.save(vision_payload, vision)
    language = tmp_path / "language.pt"
    language_payload = {
        "schema_version": 1,
        "language_contract": LANGUAGE_CONTRACT,
        "target_name": "spatial",
        "prompt": TARGET_PROMPTS["spatial"],
        "language_tokens": torch.zeros(1, 20, 2048, dtype=torch.float16),
        "attention_mask": torch.ones(1, 20, dtype=torch.bool),
        "input_ids": torch.arange(20).view(1, 20),
    }
    _attach_language_identity(language_payload, identity)
    torch.save(language_payload, language)
    config = tmp_path / "spatial.yaml"
    config.write_text(
        "\n".join(
            (
                "output_dir: runs",
                "task:",
                "  target_name: spatial",
                f"  prompt: \"{TARGET_PROMPTS['spatial']}\"",
                "data:",
                f"  manifest: {manifest.as_posix()}",
                f"  vision_cache: {vision.as_posix()}",
                f"  language_cache: {language.as_posix()}",
            )
        ),
        encoding="utf-8",
    )
    settings = load_settings(config, synthetic=True)
    payload = load_single_metric_cache(settings)
    assert payload["targets"].shape == (2,)
    assert payload["targets"].tolist() == [71.0, 52.0]
    item = LGVQSingleMetricDataset(payload, "train")[0]
    assert item["vision_tokens"].shape == (16, 49, 1024)
    assert item["quality_tokens"].shape == (16, 49, 14)
    assert item["target"].ndim == 0
    assert payload["qwen_front_identity"]["pair"]["sha256"] == identity["pair"]["sha256"]
