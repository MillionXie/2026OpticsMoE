from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from experiments.qwen3_vl_8b_multimodal_optical_weather4.datasets import (
    WEATHER4_CLASSES,
    load_weather4,
)
from experiments.qwen3_vl_8b_multimodal_optical_weather4.data_prepare import (
    _find_image_split,
    _find_label_file,
    prepare_weather_split,
)
from experiments.qwen3_vl_8b_multimodal_optical_weather4.io_utils import write_json
from experiments.qwen3_vl_8b_multimodal_optical_weather4.optics import (
    OpticalVisionBlockSurrogate,
    VisionBlockReplacement,
)
from experiments.qwen3_vl_8b_multimodal_optical_weather4.results import write_comparison
from experiments.qwen3_vl_8b_multimodal_optical_weather4.run import build_parser
from experiments.qwen3_vl_8b_multimodal_optical_weather4.settings import load_settings
from experiments.qwen3_vl_8b_multimodal_optical_weather4.modeling import MLPHead
from experiments.qwen3_vl_8b_multimodal_optical_weather4.student_training import (
    knowledge_distillation_loss,
    normalized_hidden_mse,
    train_optical_student,
)


class FakeVisionBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, cu_seqlens=None, **kwargs):
        del cu_seqlens, kwargs
        return hidden_states + self.linear(hidden_states)


class FakeVisual(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([FakeVisionBlock(hidden_size) for _ in range(3)])


class FakeQwen(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.visual = FakeVisual(hidden_size)


class FakeFullQwen(FakeQwen):
    def __init__(self, hidden_size: int = 6, language_size: int = 8) -> None:
        super().__init__(hidden_size)
        self.language_projection = nn.Linear(hidden_size, language_size)

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
        optical_layers=2,
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


def test_replacement_switches_teacher_and_student_and_captures_hidden() -> None:
    model = FakeQwen(6)
    replacement = VisionBlockReplacement(model, 2, _surrogate())
    original = replacement.original_block
    hidden = torch.randn(5, 6)
    boundaries = torch.tensor([0, 5], dtype=torch.int32)
    replacement.use_teacher()
    teacher_output = model.visual.blocks[2](hidden, cu_seqlens=boundaries)
    assert replacement.capture.input_hidden is not None
    assert replacement.capture.output_hidden is not None
    assert model.visual.blocks[2] is original
    replacement.use_student()
    student_output = model.visual.blocks[2](hidden, cu_seqlens=boundaries)
    assert student_output.shape == teacher_output.shape
    assert model.visual.blocks[2] is replacement.surrogate
    replacement.close()
    assert model.visual.blocks[2] is original


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
    loader = [(images, torch.tensor([0, 1]))]
    report = train_optical_student(
        model,
        FakeProcessor(),
        replacement,
        teacher_head,
        student_head,
        loader,
        loader,
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
    assert report["captured_shapes"]["teacher_block_output"] == [8, 6]
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


def test_config_cli_and_comparison(tmp_path: Path) -> None:
    source = Path(
        "experiments/qwen3_vl_8b_multimodal_optical_weather4/configs/"
        "bdd100k_weather4_smoke.json"
    )
    settings = load_settings(source)
    assert settings.replace_vision_block_start == 26
    assert settings.dtype == "float32"
    assert settings.attn_implementation == "eager"
    args = build_parser().parse_args(["--config", str(source), "--phase", "student_train"])
    assert args.phase == "student_train"
    assert build_parser().parse_args(
        ["--config", str(source), "--phase", "prepare_data"]
    ).phase == "prepare_data"

    metrics_dir = tmp_path / "metrics"
    write_json(
        metrics_dir / "teacher_inference.json",
        {
            "model": "teacher",
            "metrics": {"top1_accuracy": 0.8, "top5_accuracy": 1.0},
        },
    )
    write_json(
        metrics_dir / "student_inference.json",
        {
            "model": "student",
            "metrics": {"top1_accuracy": 0.7, "top5_accuracy": 1.0},
        },
    )
    comparison = write_comparison(
        tmp_path,
        "bdd100k_weather4",
        WEATHER4_CLASSES,
        settings.classification_prompt,
        {"vision_blocks": [26]},
        {"hidden": 1.0, "kd": 0.5, "ce": 0.5, "temperature": 2.0},
    )
    assert comparison["accuracy_drop"]["top1"] < 0
    assert (metrics_dir / "comparison.json").is_file()


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
