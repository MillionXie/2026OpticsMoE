import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foundation_distillation.teacher import FrozenDINOv2ImageTeacher
from common.data.foundation_distillation import teacher_input_from_student_gray


class MockDINOv2(torch.nn.Module):
    def __init__(self, tokens):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.register_buffer("tokens", tokens)

    def forward(self, pixel_values):
        return SimpleNamespace(last_hidden_state=self.tokens[: pixel_values.shape[0]])


def test_dinov2_cls_and_patch_mean_features_are_l2_normalized():
    tokens = torch.tensor(
        [
            [[3.0, 4.0], [1.0, 3.0], [3.0, 1.0]],
            [[0.0, 2.0], [2.0, 0.0], [0.0, 2.0]],
        ]
    )
    images = torch.rand(2, 3, 224, 224)

    cls_teacher = FrozenDINOv2ImageTeacher(MockDINOv2(tokens), feature_type="cls")
    cls_feature = cls_teacher(images)
    assert cls_feature.shape == (2, 2)
    assert torch.allclose(cls_feature.norm(dim=-1), torch.ones(2))
    assert torch.allclose(cls_feature[0], torch.tensor([0.6, 0.8]))

    mean_teacher = FrozenDINOv2ImageTeacher(MockDINOv2(tokens), feature_type="patch_mean")
    mean_feature = mean_teacher(images)
    expected = torch.nn.functional.normalize(tokens[:, 1:].mean(dim=1), dim=-1)
    assert mean_feature.shape == (2, 2)
    assert torch.allclose(mean_feature, expected)
    assert mean_teacher.training is False
    assert mean_teacher.model.training is False
    assert all(not parameter.requires_grad for parameter in mean_teacher.parameters())


def test_dinov2_rejects_unknown_feature_type():
    tokens = torch.ones(1, 2, 4)
    try:
        FrozenDINOv2ImageTeacher(MockDINOv2(tokens), feature_type="dense")
    except ValueError as exc:
        assert "cls" in str(exc) and "patch_mean" in str(exc)
    else:
        raise AssertionError("Unknown DINOv2 feature_type must fail.")


def test_dinov2_teacher_input_replicates_the_same_grayscale_information():
    student = torch.rand(2, 1, 120, 120)
    teacher = teacher_input_from_student_gray(
        student,
        {"type": "dinov2_image_encoder", "teacher_image_size": 224},
        normalize=False,
    )
    assert teacher.shape == (2, 3, 224, 224)
    assert torch.equal(teacher[:, 0], teacher[:, 1])
    assert torch.equal(teacher[:, 1], teacher[:, 2])
