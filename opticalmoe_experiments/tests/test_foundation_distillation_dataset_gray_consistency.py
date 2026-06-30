import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.data.foundation_distillation import teacher_input_from_student_gray


def test_teacher_uses_the_same_grayscale_information_as_student():
    student = torch.rand(2, 1, 120, 120)
    teacher = teacher_input_from_student_gray(student, normalize=False)
    assert teacher.shape == (2, 3, 224, 224)
    assert torch.equal(teacher[:, 0], teacher[:, 1])
    assert torch.equal(teacher[:, 1], teacher[:, 2])
