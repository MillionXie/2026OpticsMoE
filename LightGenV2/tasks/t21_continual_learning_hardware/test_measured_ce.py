"""CPU checks for numerical reparameterization, not new inference capacity."""
import unittest
import torch
from .refine_measured_ce import contrast_transform, fit_convex, internal_indices


class MeasuredCETest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(73)

    def test_invertible_same_parameter_count_and_exact_fold(self):
        x = torch.rand(40, 12, dtype=torch.float64) + .1
        x /= x.mean(1, keepdim=True)
        transform = contrast_transform(x)
        self.assertEqual(tuple(transform.shape), (12,12))
        self.assertEqual(int(torch.linalg.matrix_rank(transform)), 12)
        weight = torch.randn(10,12, dtype=torch.float64)
        logits = (x @ transform.T) @ weight.T
        folded = x @ (weight @ transform).T
        self.assertLess(float((logits - folded).abs().max()), 1e-10)
        self.assertLess(float(((x @ transform.T)[:, -1] - 1).abs().max()), 1e-12)

    def test_balanced_internal_TRAIN_only_split(self):
        y = torch.arange(10).repeat_interleave(100)
        fit, hold = internal_indices(y)
        self.assertEqual(len(fit),800);self.assertEqual(len(hold),200)
        self.assertFalse(set(fit.tolist()) & set(hold.tolist()))
        self.assertEqual(torch.bincount(y[hold]).tolist(),[20]*10)

    def test_convex_fit_finite_and_only_original_weight_coordinates(self):
        y = torch.arange(10).repeat(5)
        z = torch.nn.functional.one_hot(y,10).double()
        z = torch.cat((z, torch.ones(50,1,dtype=torch.float64)),1)
        weight, loss = fit_convex(z,y,.01,20)
        self.assertEqual(weight.numel(),10*z.shape[1])
        self.assertTrue(torch.isfinite(weight).all())
        self.assertLess(loss,2.3)
        self.assertEqual(int(((z @ weight.T).argmax(1)==y).sum()),50)


if __name__ == '__main__':
    unittest.main()
