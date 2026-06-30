import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foundation_distillation import teacher as teacher_module


class MockCLIP(torch.nn.Module):
    def encode_image(self, images):
        return images.mean(dim=(-2, -1))


class MockDINO(torch.nn.Module):
    def forward(self, pixel_values):
        batch = pixel_values.shape[0]
        return SimpleNamespace(last_hidden_state=torch.ones(batch, 5, 8))


def test_teacher_backend_dispatches_clip_and_dinov2(monkeypatch):
    monkeypatch.setattr(
        teacher_module,
        "_load_clip_backend",
        lambda model_name, device, requested_backend="auto": (MockCLIP(), "mock_clip"),
    )
    monkeypatch.setattr(
        teacher_module,
        "_load_dinov2_backend",
        lambda model_name, device: (
            MockDINO(),
            "transformers",
            {
                "teacher_image_size": 224,
                "teacher_image_mean": [0.485, 0.456, 0.406],
                "teacher_image_std": [0.229, 0.224, 0.225],
            },
        ),
    )
    clip_teacher, clip_info = teacher_module.load_frozen_image_teacher(
        {
            "type": "clip_image_encoder",
            "model_name": "ViT-B/32",
            "input_mode": "grayscale_replicated_rgb",
            "freeze": True,
        },
        torch.device("cpu"),
    )
    dino_teacher, dino_info = teacher_module.load_frozen_image_teacher(
        {
            "type": "dinov2_image_encoder",
            "backend": "transformers",
            "model_name": "facebook/dinov2-small",
            "feature_type": "cls",
            "input_mode": "grayscale_replicated_rgb",
            "freeze": True,
        },
        torch.device("cpu"),
    )
    assert clip_info["teacher_type"] == "clip_image_encoder"
    assert clip_info["teacher_backend"] == "mock_clip"
    assert dino_info["teacher_type"] == "dinov2_image_encoder"
    assert dino_info["teacher_backend"] == "transformers"
    assert dino_info["feature_type"] == "cls"
    assert clip_teacher(torch.rand(2, 3, 224, 224)).shape == (2, 3)
    assert dino_teacher(torch.rand(2, 3, 224, 224)).shape == (2, 8)
    compatible_teacher, compatible_backend = teacher_module.load_clip_image_encoder(
        "ViT-B/32", torch.device("cpu")
    )
    assert compatible_backend == "mock_clip"
    assert compatible_teacher(torch.rand(1, 3, 224, 224)).shape == (1, 3)


def test_unknown_teacher_type_has_clear_error():
    with pytest.raises(ValueError, match="Unknown teacher.type"):
        teacher_module.load_frozen_image_teacher(
            {
                "type": "unknown_encoder",
                "input_mode": "grayscale_replicated_rgb",
                "freeze": True,
            },
            torch.device("cpu"),
        )


def test_missing_transformers_error_is_actionable(monkeypatch):
    def missing(_model_name, _device):
        raise RuntimeError("DINOv2 teacher requires transformers.\nInstall with: pip install transformers")

    monkeypatch.setattr(teacher_module, "_load_dinov2_backend", missing)
    with pytest.raises(RuntimeError, match="pip install transformers"):
        teacher_module.load_frozen_image_teacher(
            {
                "type": "dinov2_image_encoder",
                "backend": "transformers",
                "model_name": "facebook/dinov2-small",
                "input_mode": "grayscale_replicated_rgb",
                "freeze": True,
            },
            torch.device("cpu"),
        )
