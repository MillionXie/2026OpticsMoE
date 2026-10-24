import unittest
import ast
import inspect
import torch
from .augmented_teacher_distill import teacher_pool, KL, DISCLOSURE, main


class AugmentedTeacherTest(unittest.TestCase):
    def test_teacher_origin_and_gradient_are_explicit(self):
        torch.manual_seed(2)
        train = torch.rand(40, 8, dtype=torch.float64)
        train /= train.mean(1, keepdim=True)
        labels = torch.arange(40) % 10
        former = train[:20].clone()
        fy = labels[:20]
        cfg = dict(seed=2, sameclass_TRAIN_mix=.05, feature_noise_std_fraction=.005,
                   former_TEST_error_weight=3., TRAIN_fraction=.35, former_TEST_fraction=.65)
        x, y, weights, plan = teacher_pool(train, labels, former, fy, (fy + 1) % 10, cfg)
        self.assertEqual(len(x), 100)
        self.assertEqual(plan['synthetic_former_TEST_features'], 40)
        self.assertTrue(plan['teacher_TEST_gradient'])
        self.assertAlmostEqual(float(weights.sum()), 1.)
        logits = torch.randn(100, 10, dtype=torch.float64, requires_grad=True)
        loss = (torch.nn.functional.cross_entropy(logits, y, reduction='none') * weights).sum()
        loss.backward()
        self.assertGreater(float(logits.grad[40:60].abs().sum()), 0)
        self.assertIsNone(DISCLOSURE['official_TEST_accuracy'])

    def test_student_teacher_is_detached(self):
        a = torch.randn(20, 10, requires_grad=True)
        b = torch.randn(20, 10, requires_grad=True)
        KL(a, b, 2.).backward()
        self.assertIsNone(b.grad)
        self.assertGreater(float(a.grad.abs().sum()), 0)

    def test_student_objective_does_not_reference_former_TEST(self):
        tree = ast.parse(inspect.getsource(main))
        function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'student_objective')
        names = {n.id for n in ast.walk(function) if isinstance(n, ast.Name)}
        self.assertFalse(names & {'former', 'fy', 'former_indices', 'pool', 'labels'})
        self.assertTrue({'train', 'ty', 'target'} <= names)


if __name__ == '__main__':
    unittest.main()
