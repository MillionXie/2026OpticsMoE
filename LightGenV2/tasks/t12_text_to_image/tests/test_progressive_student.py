from __future__ import annotations

import torch
from torch import nn

from LightGenV2.tasks.t12_text_to_image.progressive_student import copy_overlapping_state
from LightGenV2.tasks.t12_text_to_image.progressive_student_training import DistillationDataset


def test_overlap_copy_warm_starts_smaller_tensors() -> None:
    source = nn.Sequential(nn.Linear(8, 12), nn.Linear(12, 6))
    target = nn.Sequential(nn.Linear(5, 7), nn.Linear(7, 4))
    with torch.no_grad():
        for index, parameter in enumerate(source.parameters()):
            parameter.copy_(torch.arange(parameter.numel()).reshape_as(parameter) + index * 1000)
    report = copy_overlapping_state(target, source.state_dict())
    assert report["partial_values"] > 0
    assert report["coverage"] == 1.0
    assert torch.equal(target[0].weight, source[0].weight[:7, :5])
    assert torch.equal(target[1].weight, source[1].weight[:4, :7])


def test_distillation_dataset_preserves_attribute_index_keys(tmp_path) -> None:
    student = {
        key: torch.zeros(1, *shape)
        for key, shape in {
            "reference": (4, 2, 2), "target": (4, 2, 2),
            "foreground_mask": (1, 2, 2), "background_mask": (1, 2, 2),
            "qwen_text": (8,),
        }.items()
    }
    for index, key in enumerate(
        ("room_indices", "tone_indices", "brightness_indices", "direction_indices")
    ):
        student[key] = [index]
    teacher = {
        "prediction": torch.zeros(1, 4, 2, 2),
        "noise": torch.zeros(1, 4, 2, 2),
    }
    student_path = tmp_path / "student.pt"
    teacher_path = tmp_path / "teacher.pt"
    torch.save(student, student_path)
    torch.save(teacher, teacher_path)

    item = DistillationDataset(student_path, teacher_path)[0]

    assert item["room_indices"] == 0
    assert item["direction_indices"] == 3
