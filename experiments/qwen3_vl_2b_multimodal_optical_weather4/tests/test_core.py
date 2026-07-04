from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from experiments.qwen3_vl_2b_multimodal_optical_weather4.datasets import (
    WEATHER4_CLASSES,
    load_weather4,
    stratified_split_indices,
)
from experiments.qwen3_vl_2b_multimodal_optical_weather4.data_prepare import (
    _find_image_split,
    _find_label_file,
    prepare_weather_split,
)
from experiments.qwen3_vl_2b_multimodal_optical_weather4.io_utils import write_json
from experiments.qwen3_vl_2b_multimodal_optical_weather4.optics import (
    OpticalVisionBlockSurrogate,
    VisionBlockReplacement,
)
from experiments.qwen3_vl_2b_multimodal_optical_weather4.results import write_comparison
from experiments.qwen3_vl_2b_multimodal_optical_weather4.run import (
    _build_replacement,
    _teacher_feature_metadata,
    build_parser,
)
from experiments.qwen3_vl_2b_multimodal_optical_weather4.settings import load_settings
from experiments.qwen3_vl_2b_multimodal_optical_weather4.modeling import MLPHead
from experiments.qwen3_vl_2b_multimodal_optical_weather4.student_training import (
    knowledge_distillation_loss,
    normalized_hidden_mse,
    train_optical_student,
)
from experiments.qwen3_vl_2b_multimodal_optical_weather4.teacher_cache import (
    CachedTeacherDataset,
    TeacherCacheStore,
    build_teacher_cache,
    expected_teacher_cache_metadata,
    make_cached_teacher_loader,
    validate_teacher_cache,
)


class FakeVisionBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, cu_seqlens=None, **kwargs):
        del cu_seqlens, kwargs
        return hidden_states + self.linear(hidden_states)


class LabelOnlyDataset:
    def __init__(self, counts: list[int]) -> None:
        self.labels = [label for label, count in enumerate(counts) for _ in range(count)]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return None, self.labels[index]


class FakeVisual(nn.Module):
    def __init__(self, hidden_size: int, block_count: int = 3) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [FakeVisionBlock(hidden_size) for _ in range(block_count)]
        )


class FakeQwen(nn.Module):
    def __init__(self, hidden_size: int, block_count: int = 3) -> None:
        super().__init__()
        self.visual = FakeVisual(hidden_size, block_count)


class FakeFullQwen(FakeQwen):
    def __init__(self, hidden_size: int = 6, language_size: int = 8) -> None:
        super().__init__(hidden_size)
        self.language_projection = nn.Linear(hidden_size, language_size)
        self.config = SimpleNamespace(
            vision_config=SimpleNamespace(depth=3, hidden_size=hidden_size),
            text_config=SimpleNamespace(hidden_size=language_size),
        )

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw, **kwargs):
        del image_grid_thw, kwargs
        batch_size = input_ids.shape[0]
        tokens_per_image = pixel_values.shape[0] // batch_size
        boundaries = torch.arange(
            0,
            (batch_size + 1) * tokens_per_image,
            tokens_per_image,
            device=pixel_values.device,
            dtype=torch.int32,
        )
        hidden = pixel_values
        for block in self.visual.blocks:
            hidden = block(hidden, cu_seqlens=boundaries)
        pooled = hidden.reshape(batch_size, tokens_per_image, -1).mean(dim=1)
        answer = self.language_projection(pooled)
        sequence = answer.unsqueeze(1).expand(-1, input_ids.shape[1], -1)
        return type("Output", (), {"hidden_states": (sequence,)})()


class FakeProcessor:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        del tokenize, add_generation_prompt
        return messages[0]["content"][1]["text"]

    def __call__(self, text, images, return_tensors, padding):
        del text, return_tensors, padding
        batch_size = len(images)
        return {
            "input_ids": torch.ones(batch_size, 3, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, 3, dtype=torch.long),
            "pixel_values": torch.randn(batch_size * 4, 6),
            "image_grid_thw": torch.tensor([[1, 2, 2]] * batch_size),
        }


def _surrogate(hidden_size: int = 6) -> OpticalVisionBlockSurrogate:
    return OpticalVisionBlockSurrogate(
        hidden_size=hidden_size,
        optical_dim=4,
        optical_layers=1,
        optical_field_size=8,
        optical_padding_size=12,
        wavelength_nm=532.0,
        pixel_pitch_um=17.0,
        mask_distance_cm=5.0,
    )


def test_optical_surrogate_preserves_packed_shape_and_gradients() -> None:
    module = _surrogate()
    hidden = torch.randn(10, 6, requires_grad=True)
    cu_seqlens = torch.tensor([0, 4, 10], dtype=torch.int32)
    output = module(hidden, cu_seqlens=cu_seqlens)
    assert output.shape == hidden.shape
    output.square().mean().backward()
    assert module.input_adapter.weight.grad is not None
    assert module.optical_group.phase_masks.grad is not None
    assert module.output_adapter.weight.grad is not None


def test_optical_surrogate_accepts_bfloat16_backbone_boundary() -> None:
    module = _surrogate()
    hidden = torch.randn(8, 6, dtype=torch.bfloat16, requires_grad=True)
    output = module(hidden, cu_seqlens=torch.tensor([0, 4, 8], dtype=torch.int32))
    assert output.dtype == torch.bfloat16
    assert output.shape == hidden.shape
    output.float().square().mean().backward()
    assert module.optical_group.phase_masks.grad is not None


def test_optical_surrogate_has_no_electronic_residual_bypass() -> None:
    module = _surrogate()
    nn.init.zeros_(module.output_adapter.weight)
    nn.init.zeros_(module.output_adapter.bias)
    hidden = torch.randn(8, 6)
    output = module(hidden, cu_seqlens=torch.tensor([0, 4, 8], dtype=torch.int32))
    assert torch.count_nonzero(output) == 0


def test_packed_optical_input_requires_per_image_boundaries() -> None:
    with pytest.raises(RuntimeError, match="cu_seqlens"):
        _surrogate()(torch.randn(8, 6))


def test_replacement_switches_teacher_and_student_and_captures_hidden() -> None:
    model = FakeQwen(6, block_count=9)
    groups = [(1, 4), (5, 8)]
    replacement = VisionBlockReplacement(model, groups, [_surrogate(), _surrogate()])
    originals = list(model.visual.blocks)
    hidden = torch.randn(5, 6)
    boundaries = torch.tensor([0, 5], dtype=torch.int32)
    replacement.use_teacher()
    teacher_output = hidden
    for block in model.visual.blocks:
        teacher_output = block(teacher_output, cu_seqlens=boundaries)
    assert all(capture.input_hidden is not None for capture in replacement.captures)
    assert all(capture.output_hidden is not None for capture in replacement.captures)
    assert list(model.visual.blocks) == originals
    replacement.use_student()
    student_output = hidden
    for block in model.visual.blocks:
        student_output = block(student_output, cu_seqlens=boundaries)
    assert student_output.shape == teacher_output.shape
    assert model.visual.blocks[1] is replacement.surrogates[0]
    assert model.visual.blocks[5] is replacement.surrogates[1]
    assert all(
        model.visual.blocks[index].__class__.__name__ == "VisionBlockBypass"
        for index in (2, 3, 4, 6, 7, 8)
    )
    replacement.close()
    assert list(model.visual.blocks) == originals


def test_distillation_losses_are_differentiable() -> None:
    student_hidden = torch.randn(2, 4, 6, requires_grad=True)
    teacher_hidden = torch.randn(2, 4, 6)
    student_logits = torch.randn(2, 4, requires_grad=True)
    teacher_logits = torch.randn(2, 4)
    loss = normalized_hidden_mse(student_hidden, teacher_hidden)
    loss = loss + knowledge_distillation_loss(student_logits, teacher_logits, 2.0)
    loss.backward()
    assert student_hidden.grad is not None
    assert student_logits.grad is not None


def test_joint_student_training_keeps_gradient_path(tmp_path: Path) -> None:
    model = FakeFullQwen()
    model.requires_grad_(False)
    replacement = VisionBlockReplacement(model, 2, _surrogate())
    teacher_head = MLPHead(8, 5, 4, 0.0)
    student_head = MLPHead(8, 5, 4, 0.0)
    student_head.load_state_dict(teacher_head.state_dict())
    images = [Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))]
    validation_loader = [(images, torch.tensor([0, 1]))]
    train_loader = [
        {
            "images": images,
            "labels": torch.tensor([0, 1]),
            "sample_indices": torch.tensor([0, 1]),
            "teacher_logits": torch.randn(2, 4),
            "teacher_answer_hidden": torch.randn(2, 8),
            "teacher_token_counts": [4, 4],
            "teacher_group_outputs": [torch.randn(8, 6)],
            "group_names": ["group_2_2"],
        }
    ]
    report = train_optical_student(
        model,
        FakeProcessor(),
        replacement,
        student_head,
        train_loader,
        validation_loader,
        WEATHER4_CLASSES,
        "Classify weather. Answer:",
        torch.device("cpu"),
        tmp_path,
        epochs=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        temperature=2.0,
        hidden_weight=1.0,
        kd_weight=0.5,
        ce_weight=0.5,
        progress=False,
    )
    assert report["best_epoch"] == 1
    groups = report["captured_shapes"]["distillation_groups"]
    assert groups[0]["teacher_output_shape"] == [8, 6]
    assert report["teacher_forward_during_student_training"] is False
    checkpoint = torch.load(
        tmp_path / "checkpoints" / "optical_surrogate.pt", weights_only=True
    )
    assert checkpoint["electronic_residual_bypass"] is False
    assert (tmp_path / "checkpoints" / "student_mlp.pt").is_file()
    assert (tmp_path / "checkpoints" / "optical_surrogate.pt").is_file()
    replacement.close()


def test_weather4_imagefolder_order_and_balanced_limit(tmp_path: Path) -> None:
    root = tmp_path / "bdd100k_weather4"
    for split in ("train", "test"):
        for class_name in WEATHER4_CLASSES:
            directory = root / split / class_name
            directory.mkdir(parents=True)
            for index in range(2):
                Image.new("RGB", (8, 8), color=(index * 20, 0, 0)).save(
                    directory / f"{index}.png"
                )
    bundle = load_weather4(
        root,
        resize_to=None,
        train_limit=None,
        test_limit=None,
        train_limit_per_class=1,
        test_limit_per_class=1,
        imagefolder_train="train",
        imagefolder_test="test",
        seed=42,
        download=True,
    )
    assert bundle.class_names == WEATHER4_CLASSES
    assert len(bundle.train) == len(bundle.test) == 4
    assert sorted(bundle.train[index][1] for index in range(4)) == [0, 1, 2, 3]


def test_full_weather4_stratified_split_has_expected_sizes() -> None:
    dataset = LabelOnlyDataset([37344, 5070, 5549, 130])
    train_indices, validation_indices = stratified_split_indices(dataset, 0.1, 42)
    assert len(train_indices) == 43284
    assert len(validation_indices) == 4809


def test_config_cli_and_comparison(tmp_path: Path) -> None:
    source = Path(
        "experiments/qwen3_vl_2b_multimodal_optical_weather4/configs/"
        "bdd100k_weather4_smoke.json"
    )
    settings = load_settings(source)
    assert settings.model_id == "Qwen/Qwen3-VL-2B-Instruct"
    assert settings.replace_last_n_vision_blocks == 20
    assert settings.optical_conversions == 5
    assert settings.teacher_blocks_per_conversion == 4
    assert settings.optical_layers == 1
    assert settings.dtype == "float32"
    assert settings.attn_implementation == "eager"
    args = build_parser().parse_args(["--config", str(source), "--phase", "student_train"])
    assert args.phase == "student_train"
    assert build_parser().parse_args(
        ["--config", str(source), "--phase", "prepare_data"]
    ).phase == "prepare_data"
    assert build_parser().parse_args(
        ["--config", str(source), "--phase", "teacher_cache"]
    ).phase == "teacher_cache"

    metrics_dir = tmp_path / "metrics"
    write_json(
        metrics_dir / "teacher_inference.json",
        {
            "model": "teacher",
            "metrics": {
                "top1_accuracy": 0.8,
                "top5_accuracy": 1.0,
                "per_class_accuracy": {"clear": 0.9, "foggy": 0.2},
            },
        },
    )
    write_json(
        metrics_dir / "student_inference.json",
        {
            "model": "student",
            "metrics": {
                "top1_accuracy": 0.7,
                "top5_accuracy": 1.0,
                "per_class_accuracy": {"clear": 0.8, "foggy": 0.1},
            },
        },
    )
    comparison = write_comparison(
        tmp_path,
        "bdd100k_weather4",
        WEATHER4_CLASSES,
        settings.classification_prompt,
        {"vision_blocks": [26]},
        {"hidden": 1.0, "kd": 0.5, "ce": 0.5, "temperature": 2.0},
        {"train_counts": {"clear": 10, "foggy": 1}},
    )
    assert comparison["accuracy_drop"]["top1"] > 0
    assert comparison["teacher"]["per_class_accuracy"]["foggy"] == 0.2
    assert comparison["class_imbalance_counts"]["train_counts"]["foggy"] == 1
    assert (metrics_dir / "comparison.json").is_file()


def test_runtime_depth_builds_last_twenty_block_groups() -> None:
    model = FakeQwen(6, block_count=24)
    model.config = SimpleNamespace(
        vision_config=SimpleNamespace(depth=24, hidden_size=6),
        text_config=SimpleNamespace(hidden_size=8),
    )
    source = Path(
        "experiments/qwen3_vl_2b_multimodal_optical_weather4/configs/"
        "bdd100k_weather4_smoke.json"
    )
    settings = load_settings(source)
    replacement = _build_replacement(
        SimpleNamespace(model=model), settings, torch.device("cpu")
    )
    assert replacement.block_groups == [
        (4, 7),
        (8, 11),
        (12, 15),
        (16, 19),
        (20, 23),
    ]
    replacement.close()


def test_teacher_group_cache_is_sharded_validated_and_batch_aligned(
    tmp_path: Path,
) -> None:
    model = FakeFullQwen()
    replacement = VisionBlockReplacement(model, 2, _surrogate())
    head = MLPHead(8, 5, 4, 0.0).eval()
    images = [Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))]
    labels = torch.tensor([0, 1])
    settings = SimpleNamespace(
        output_dir=tmp_path,
        model_id="Qwen/Qwen3-VL-2B-Instruct",
        classification_prompt="Classify weather. Answer:",
        data_root=tmp_path / "data",
        processor_min_pixels=50176,
        processor_max_pixels=50176,
        resize_to=None,
        dtype="float32",
        attn_implementation="eager",
        cache_dtype="float16",
        teacher_cache_shard_size=1,
        progress=False,
    )
    (tmp_path / "checkpoints").mkdir()
    torch.save({"state_dict": head.state_dict()}, tmp_path / "checkpoints" / "teacher_mlp.pt")
    manifest_path = build_teacher_cache(
        split="train",
        model=model,
        processor=FakeProcessor(),
        replacement=replacement,
        teacher_head=head,
        loader=[(images, labels)],
        dataset_size=2,
        class_names=WEATHER4_CLASSES,
        settings=settings,
        device=torch.device("cpu"),
    )
    expected = expected_teacher_cache_metadata(
        split="train",
        samples=2,
        settings=settings,
        model=model,
        replacement=replacement,
        class_names=WEATHER4_CLASSES,
    )
    assert validate_teacher_cache(manifest_path, expected) == (True, [])
    manifest = torch.load(manifest_path, weights_only=True)
    assert len(manifest["shards"]) == 2
    assert manifest["metadata"]["image_grid_thw_summary"]["values"]
    first_shard = torch.load(
        manifest_path.parent / manifest["shards"][0]["path"], weights_only=True
    )
    assert "input" in first_shard["groups"]["group_2_2"]
    assert "output" in first_shard["groups"]["group_2_2"]

    store = TeacherCacheStore(manifest_path, expected)
    cached = CachedTeacherDataset(list(zip(images, labels.tolist())), store)
    loader = make_cached_teacher_loader(cached, [0, 1], 2, 0, 42)
    batches = list(loader)
    assert sum(len(batch["labels"]) for batch in batches) == 2
    assert [count for batch in batches for count in batch["teacher_token_counts"]] == [4, 4]
    assert sum(batch["teacher_group_outputs"][0].shape[0] for batch in batches) == 8
    assert sorted(label for batch in batches for label in batch["labels"].tolist()) == [0, 1]

    stale = dict(expected)
    stale["processor_max_pixels"] = 99999
    valid, changed = validate_teacher_cache(manifest_path, stale)
    assert valid is False
    assert "processor_max_pixels" in changed
    replacement.close()


def test_teacher_feature_metadata_covers_semantic_inputs(tmp_path: Path) -> None:
    source = Path(
        "experiments/qwen3_vl_2b_multimodal_optical_weather4/configs/"
        "bdd100k_weather4_smoke.json"
    )
    settings = load_settings(source)
    settings.data_root = tmp_path / "data"
    data = type(
        "Data",
        (),
        {"train": list(range(4)), "test": list(range(2)), "class_names": WEATHER4_CLASSES},
    )()
    metadata = _teacher_feature_metadata("train", data, settings)
    expected = {
        "cache_schema_version",
        "num_classes",
        "class_names",
        "data_root",
        "resize_to",
        "processor_min_pixels",
        "processor_max_pixels",
        "dtype",
        "attn_implementation",
    }
    assert expected <= metadata.keys()


def test_prepare_weather_split_from_bdd_labels(tmp_path: Path) -> None:
    images = tmp_path / "raw" / "train"
    images.mkdir(parents=True)
    labels = []
    for index, weather in enumerate([*WEATHER4_CLASSES, "overcast"]):
        name = f"image-{index}.jpg"
        Image.new("RGB", (8, 8)).save(images / name)
        labels.append({"name": name, "attributes": {"weather": weather}})
    labels_file = tmp_path / "det_train.json"
    labels_file.write_text(json.dumps(labels), encoding="utf-8")

    report = prepare_weather_split(images, labels_file, tmp_path / "prepared")

    assert report["total"] == 4
    assert report["ignored_non_weather4"] == 1
    for weather in WEATHER4_CLASSES:
        assert (tmp_path / "prepared" / weather).is_dir()
        assert len(list((tmp_path / "prepared" / weather).iterdir())) == 1


def test_finds_kaggle_v2_label_names(tmp_path: Path) -> None:
    labels = tmp_path / "labels" / "det_v2_train_release.json"
    labels.parent.mkdir()
    labels.write_text("[]", encoding="utf-8")
    assert _find_label_file(tmp_path, "train") == labels


def test_prefers_kaggle_100k_images_over_10k(tmp_path: Path) -> None:
    base = tmp_path / "bdd100k" / "bdd100k" / "images"
    for size in ("10k", "100k"):
        directory = base / size / "train"
        directory.mkdir(parents=True)
        Image.new("RGB", (8, 8)).save(directory / f"{size}.jpg")
    assert _find_image_split(tmp_path, "train") == base / "100k" / "train"
