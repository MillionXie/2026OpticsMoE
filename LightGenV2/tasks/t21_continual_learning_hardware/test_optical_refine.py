import unittest
import torch

from .refine_native_optics import TRAIN_objective


class NativeOpticalRefineTest(unittest.TestCase):
    def test_teacher_is_detached_and_student_learns(self):
        student = torch.randn(8, 10, requires_grad=True)
        teacher = torch.randn(8, 10, requires_grad=True)
        labels = torch.arange(8)
        cfg = dict(label_smoothing=.02, distillation_temperature=2., distillation_weight=.5)
        loss = TRAIN_objective(student, labels, teacher, cfg)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(student.grad.abs().sum()), 0)
        self.assertIsNone(teacher.grad)


if __name__ == '__main__':
    unittest.main()
