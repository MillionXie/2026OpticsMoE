from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from experiments.qwen3_vl_2b_flickr30k_image_text_matching_multimodal_homogeneous_moe9.datasets import load_flickr30k
from experiments.qwen3_vl_2b_flickr30k_image_text_matching_multimodal_homogeneous_moe9.features import (
    pool_answer_hidden_state, preprocess_image_text,
)
from experiments.qwen3_vl_2b_flickr30k_image_text_matching_multimodal_homogeneous_moe9.metrics import binary_classification_metrics
from experiments.qwen3_vl_2b_flickr30k_image_text_matching_multimodal_homogeneous_moe9.modeling import build_head
from experiments.qwen3_vl_2b_flickr30k_image_text_matching_multimodal_homogeneous_moe9.optics.geometry import MoEGeometry
from experiments.qwen3_vl_2b_flickr30k_image_text_matching_multimodal_homogeneous_moe9.optics.moe import (
    HomogeneousMoEOpticalCore, LanguageDeepStackHomogeneousMoE,
)
from experiments.qwen3_vl_2b_flickr30k_image_text_matching_multimodal_homogeneous_moe9.processor_cache import (
    collate_processor_samples, expected_processor_metadata,
)
from experiments.qwen3_vl_2b_flickr30k_image_text_matching_multimodal_homogeneous_moe9.settings import Settings, load_settings
from experiments.qwen3_vl_2b_flickr30k_image_text_matching_multimodal_homogeneous_moe9.teacher_cache import (
    _AsyncShardWriter, iter_cached_input_batches, load_teacher_logits, pack_teacher_rows,
    packed_teacher_targets_to_cpu, write_teacher_logits,
)


ROOT = Path("experiments/qwen3_vl_2b_flickr30k_image_text_matching_multimodal_homogeneous_moe9")
OPTICAL = ROOT / "configs" / "flickr30k_itm_vision_language_optical_smoke.json"
ELECTRONIC = ROOT / "configs" / "flickr30k_itm_vision_electronic_language_smoke.json"
FORMAL = ROOT / "configs" / "flickr30k_itm_vision_language_optical.json"


class FakeSplit:
    def __init__(self, rows: list[dict], columns: list[str] | None = None) -> None:
        self.rows = rows; self.column_names = columns or list(rows[0]); self._fingerprint = "fake-fingerprint"
    def __len__(self): return len(self.rows)
    def __getitem__(self, index): return {key: self.rows[index][key] for key in self.column_names}
    def select_columns(self, columns): return FakeSplit(self.rows, list(columns))


def _fake_flickr() -> dict[str, FakeSplit]:
    rows = []
    counts = {"train": 6, "val": 3, "test": 4}
    number = 0
    for split, count in counts.items():
        for _ in range(count):
            image_id = f"image-{number}"
            rows.append({"image": Image.new("RGB", (8, 8), (number, 3, 4)),
                         "caption": [f"unique caption {number} number {slot}" for slot in range(5)],
                         "split": split, "img_id": image_id, "filename": f"{image_id}.jpg"})
            number += 1
    return {"test": FakeSplit(rows)}


def _settings(tmp_path: Path) -> Settings:
    settings = Settings(data_root=tmp_path / "cache", output_dir=tmp_path / "run",
                        validate_standard_counts=False, num_workers=0, train_image_limit=None,
                        test_image_limit=None)
    settings.validate(); return settings


def test_configs_select_two_language_modes_and_no_validation() -> None:
    optical = load_settings(OPTICAL); electronic = load_settings(ELECTRONIC)
    assert optical.student_language_mode == "optical_moe"
    assert electronic.student_language_mode == "electronic"
    assert optical.student_selection_split == "test" and optical.student_selection_metric == "auroc"
    assert optical.max_visual_tokens == optical.max_language_tokens == 120
    assert optical.processor_min_pixels == optical.processor_max_pixels == 20480
    assert optical.prompt_template.count("{caption}") == 1
    assert load_settings(FORMAL).feature_batch_size == 32


def test_fixed_pair_manifest_is_balanced_deterministic_and_leak_free(tmp_path: Path) -> None:
    settings = _settings(tmp_path); source = _fake_flickr()
    first = load_flickr30k(settings, raw_dataset=source)
    digest = first.metadata["pair_manifest_digests"]
    second_settings = _settings(tmp_path); second = load_flickr30k(second_settings, raw_dataset=source)
    assert second.metadata["pair_manifest_digests"] == digest
    assert len(first.train) == 12 and len(first.test) == 8
    train_images = {pair.image_id for pair in first.train.pairs}; test_images = {pair.image_id for pair in first.test.pairs}
    assert train_images.isdisjoint(test_images)
    for dataset in (first.train, first.test):
        assert sum(pair.label == 1 for pair in dataset.pairs) == sum(pair.label == 0 for pair in dataset.pairs)
        for pair in dataset.pairs:
            if pair.label == 0:
                assert pair.caption_source_image_id != pair.image_id
                assert pair.caption not in dataset.images[pair.image_id].captions
    assert (settings.output_dir / "pair_manifests" / "train.jsonl").is_file()


def test_manifest_or_prompt_change_rejects_stale_pairing(tmp_path: Path) -> None:
    source = _fake_flickr(); settings = _settings(tmp_path); load_flickr30k(settings, raw_dataset=source)
    changed = _settings(tmp_path); changed.prompt_template = "Different prompt: {caption}"
    with pytest.raises(RuntimeError, match="manifest mismatch"):
        load_flickr30k(changed, raw_dataset=source)


class FakeProcessor:
    def __init__(self): self.texts = []
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert not tokenize and add_generation_prompt
        return messages[0]["content"][1]["text"]
    def __call__(self, *, text, images, padding, return_tensors):
        self.texts = list(text)
        return {"input_ids": torch.ones(len(text), 3, dtype=torch.long),
                "attention_mask": torch.ones(len(text), 3, dtype=torch.long),
                "pixel_values": torch.ones(len(text), 4),
                "image_grid_thw": torch.tensor([[1, 1, 1]] * len(text))}


def test_preprocess_accepts_per_sample_prompts() -> None:
    processor = FakeProcessor(); images = [Image.new("RGB", (4, 4)), Image.new("RGB", (4, 4))]
    output = preprocess_image_text(processor, images, ["caption A", "caption B"])
    assert processor.texts == ["caption A", "caption B"] and output["input_ids"].shape == (2, 3)
    with pytest.raises(ValueError, match="length mismatch"):
        preprocess_image_text(processor, images, ["only one"])


def _encoder(hidden_size: int = 8, max_tokens: int = 120) -> HomogeneousMoEOpticalCore:
    module = HomogeneousMoEOpticalCore.__new__(HomogeneousMoEOpticalCore); nn.Module.__init__(module)
    module.hidden_size = hidden_size; module.max_tokens = max_tokens; module.geometry = MoEGeometry()
    module.input_adapter = nn.Linear(hidden_size, 120); module.input_norm = nn.LayerNorm(120); module.nonnegative = nn.Softplus()
    return module


def test_optical_token_rows_are_nonnegative_zero_padded_and_strict() -> None:
    encoder = _encoder(); field = encoder.encode_groups([torch.randn(60, 8)])
    assert field.shape == (1, 120, 120) and torch.all(field >= 0) and torch.count_nonzero(field[:, 60:]) == 0
    with pytest.raises(RuntimeError, match="visual token count 121"):
        encoder.encode_groups([torch.randn(121, 8)])


def test_language_overflow_is_explicit() -> None:
    language = LanguageDeepStackHomogeneousMoE.__new__(LanguageDeepStackHomogeneousMoE); nn.Module.__init__(language)
    language.core = SimpleNamespace(max_tokens=120)
    with pytest.raises(RuntimeError, match="language sequence length 121"):
        language.set_attention_mask(torch.ones(1, 121))


def test_binary_head_outputs_raw_logits_and_backward() -> None:
    settings = load_settings(OPTICAL); head = build_head(settings, 2048)
    logits = head(torch.randn(4, 2048)); assert logits.shape == (4,)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1., 0., 1.]))
    loss.backward(); assert all(parameter.grad is not None for parameter in head.parameters())
    assert head.specification()["output"] == "raw_logit"


def test_binary_metrics_perfect_and_balanced() -> None:
    report = binary_classification_metrics([0, 1, 0, 1], [-5, 5, -2, 3])
    assert report["accuracy"] == report["balanced_accuracy"] == report["auroc"] == 1.0
    assert report["confusion_matrix"] == [[2, 0], [0, 2]]
    assert report["positive_samples"] == report["negative_samples"] == 2


def test_rolling_single_class_metrics_are_explicitly_undefined_not_fatal() -> None:
    report = binary_classification_metrics([0], [-0.2])
    assert np.isnan(report["auroc"]) and np.isnan(report["average_precision"])


def test_teacher_logit_cache_is_raw_and_identity_checked(tmp_path: Path) -> None:
    settings = _settings(tmp_path); settings.pair_manifest_digests = {"train": "digest"}
    path = write_teacher_logits(settings.output_dir, "train", torch.tensor([-3.0, 2.5]), settings)
    loaded = load_teacher_logits(path, settings, "train")
    assert loaded.tolist() == [-3.0, 2.5]
    settings.pair_manifest_digests["train"] = "different"
    with pytest.raises(RuntimeError, match="logit cache mismatch"):
        load_teacher_logits(path, settings, "train")


def test_processor_cache_identity_contains_pair_manifest(tmp_path: Path) -> None:
    settings = _settings(tmp_path); settings.pair_manifest_digests = {"train": "abc"}
    metadata = expected_processor_metadata("train", 12, settings)
    assert metadata["pair_manifest_digest"] == "abc"
    assert metadata["prompt_template"] == settings.prompt_template
    assert metadata["negative_sampling_algorithm"] == settings.negative_sampling_algorithm


def test_answer_position_uses_last_non_padding_token() -> None:
    hidden = torch.arange(2 * 4 * 3).reshape(2, 4, 3).float()
    mask = torch.tensor([[0, 1, 1, 0], [1, 1, 1, 1]])
    answer, positions = pool_answer_hidden_state(hidden, mask)
    assert positions.tolist() == [2, 3] and torch.equal(answer[0], hidden[0, 2])


def test_cached_multimodal_batch_padding_and_pixels() -> None:
    rows = [{"input_ids": torch.tensor([1, 2]), "sequence_length": 2,
             "pixel_values": torch.ones(3, 4), "image_grid_thw": torch.tensor([1, 1, 3])},
            {"input_ids": torch.tensor([3, 4, 5]), "sequence_length": 3,
             "pixel_values": torch.ones(2, 4), "image_grid_thw": torch.tensor([1, 1, 2])}]
    batch = collate_processor_samples(rows, {"padding_side": "left", "pad_token_id": 0})
    assert batch["input_ids"].tolist() == [[0, 1, 2], [3, 4, 5]]
    assert batch["pixel_values"].shape == (5, 4)


def test_teacher_batches_reuse_exact_processor_cache() -> None:
    rows = [{"input_ids": torch.tensor([1, 2]), "sequence_length": 2,
             "pixel_values": torch.ones(4, 3), "image_grid_thw": torch.tensor([1, 2, 2])},
            {"input_ids": torch.tensor([3, 4, 5]), "sequence_length": 3,
             "pixel_values": torch.full((4, 3), 2.0), "image_grid_thw": torch.tensor([1, 2, 2])},
            {"input_ids": torch.tensor([6]), "sequence_length": 1,
             "pixel_values": torch.full((4, 3), 3.0), "image_grid_thw": torch.tensor([1, 2, 2])}]

    class Store:
        metadata = {"padding_side": "left", "pad_token_id": 0}
        def __len__(self): return len(rows)
        def get(self, index): return rows[index]

    batches = list(iter_cached_input_batches(Store(), [0.0, 1.0, 0.0], 2))
    assert len(batches) == 2
    inputs, labels, indices = batches[0]
    assert inputs["input_ids"].tolist() == [[0, 1, 2], [3, 4, 5]]
    assert inputs["pixel_values"].shape == (8, 3)
    assert labels.tolist() == [0.0, 1.0] and indices.tolist() == [0, 1]
    assert batches[1][2].tolist() == [2]


def test_teacher_targets_are_split_after_packed_transfer() -> None:
    answer = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    packed = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    answer_cpu, groups = packed_teacher_targets_to_cpu(answer, {7: packed}, [7], [2, 3], torch.float16)
    assert answer_cpu.device.type == "cpu" and answer_cpu.dtype == torch.float16
    assert [tuple(group.shape) for group in groups[7]] == [(2, 3), (3, 3)]
    assert torch.equal(groups[7][0].float(), packed[:2])
    assert torch.equal(groups[7][1].float(), packed[2:])


def test_teacher_shard_packs_variable_token_taps_with_offsets() -> None:
    rows = []
    for index, count in enumerate((2, 3)):
        rows.append({"sample_index": index, "label": float(index),
                     "image_grid_thw": torch.tensor([1, 2, count]),
                     "visual_token_count": count, "sequence_length": 4 + index,
                     "teacher_answer_hidden": torch.full((4,), float(index)),
                     "teacher_vision_taps": [torch.full((count, 3), float(index + tap)) for tap in range(4)]})
    payload = pack_teacher_rows(rows)
    assert payload["visual_token_offsets"].tolist() == [0, 2, 5]
    assert len(payload["teacher_vision_taps"]) == 4
    assert all(tuple(tap.shape) == (5, 3) for tap in payload["teacher_vision_taps"])
    assert torch.equal(payload["teacher_vision_taps"][2][:2], rows[0]["teacher_vision_taps"][2])
    assert torch.equal(payload["teacher_vision_taps"][2][2:5], rows[1]["teacher_vision_taps"][2])


def test_teacher_shard_writer_is_bounded_and_does_not_reread_for_hash(tmp_path: Path) -> None:
    rows = [{"sample_index": 0, "label": 1.0,
             "image_grid_thw": torch.tensor([1, 2, 2]),
             "visual_token_count": 2, "sequence_length": 4,
             "teacher_answer_hidden": torch.ones(4),
             "teacher_vision_taps": [torch.ones(2, 3) for _ in range(4)]}]
    writer = _AsyncShardWriter(tmp_path, max_pending=2)
    writer.submit(rows)
    records = writer.finish()
    assert len(records) == 1 and records[0]["count"] == 1
    assert "sha256" not in records[0]
    payload = torch.load(records[0]["path"], map_location="cpu", weights_only=True)
    assert payload["visual_token_offsets"].tolist() == [0, 2]
