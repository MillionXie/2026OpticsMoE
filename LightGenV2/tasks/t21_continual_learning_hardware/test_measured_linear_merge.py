import unittest
import torch
from .merge_measured_linear import merge_linear


class LinearMergeTests(unittest.TestCase):
    def test_equal_scores_fold_into_one_original_linear(self):
        torch.manual_seed(8)
        x = torch.randn(1000, 12); weights = [torch.randn(10, 12) for _ in range(4)]
        merged, scales = merge_linear(weights, x)
        expected = sum((x @ w.T - (x @ w.T).mean(1, keepdim=True)) / s / 4 for w, s in zip(weights, scales))
        self.assertTrue(torch.allclose(x @ merged.T, expected, atol=1e-6, rtol=1e-5))
        self.assertEqual(merged.shape, weights[0].shape)

    def test_member_scale_and_common_class_offsets_do_not_change_merge(self):
        torch.manual_seed(9)
        x = torch.randn(1000, 12); weights = [torch.randn(10, 12) for _ in range(4)]
        expected, _ = merge_linear(weights, x)
        shifted = [w * (i + 1) + torch.randn(1, 12) for i, w in enumerate(weights)]
        actual, _ = merge_linear(shifted, x)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-5))

    def test_invalid_member_count_or_degenerate_scale_rejected(self):
        x = torch.ones(1000, 12); w = torch.zeros(10, 12)
        with self.assertRaises(ValueError): merge_linear([w] * 3, x)
        with self.assertRaises(ValueError): merge_linear([w] * 4, x)


if __name__ == '__main__': unittest.main()
