import copy
import json
from pathlib import Path
import unittest

from .fullk_refine import choose, kd_loss, validate


class FullKRefinementTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((Path(__file__).parent / 'configs/kather_n49_fullk_refine_20261027.json').read_text(encoding='utf-8'))

    def test_budget_and_gpu_boundary(self):
        validate(self.cfg)
        for key, value in [('epochs', 60), ('gpu_uuid', 'another GPU'), ('patience', 30)]:
            bad = copy.deepcopy(self.cfg)
            bad[key] = value
            with self.assertRaises(ValueError):
                validate(bad)

    def test_validation_choice_ignores_test(self):
        a = {'val': {'accuracy': .78, 'macro_nll': .6}, 'test': 1.}
        b = {'val': {'accuracy': .79, 'macro_nll': .7}, 'test': 0.}
        best, rollout = choose([a, b], .78)
        self.assertIs(best, b)
        self.assertTrue(rollout)
        self.assertFalse(choose([a], .78)[1])

    def test_distillation_protects_student_from_wrong_or_less_confident_teacher(self):
        import torch
        logits = torch.tensor([[0., 3., 0.], [0., 0., 2.]], requires_grad=True)
        teacher = torch.tensor([[.1, .8, .1], [.1, .2, .7]])
        loss, fraction = kd_loss(logits.softmax(1), teacher, torch.tensor([1, 1]), 2.)
        loss.backward()
        self.assertEqual(float(loss), 0.)
        self.assertEqual(float(fraction), 0.)
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_correct_more_confident_teacher_gives_finite_learning_signal(self):
        import torch
        logits = torch.tensor([[1., 0., 0.]], requires_grad=True)
        loss, fraction = kd_loss(logits.softmax(1), torch.tensor([[.1, .8, .1]]), torch.tensor([1]), 2.)
        loss.backward()
        self.assertEqual(float(fraction), 1.)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertLess(float(logits.grad[0, 1]), 0.)


if __name__ == '__main__':
    unittest.main()
