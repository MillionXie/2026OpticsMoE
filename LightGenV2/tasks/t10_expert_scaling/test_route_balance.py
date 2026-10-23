"""CPU checks of load-aware routing gradients and batch semantics."""
import unittest

import torch

from .diagnose_topk import load_balance_loss


class RouteBalanceTests(unittest.TestCase):
    def test_balanced_load_has_unit_baseline_and_no_preferred_logit(self):
        logits = torch.zeros(4, 4, requires_grad=True)
        probability = logits.softmax(1)
        mask = 1-torch.eye(4)
        loss = load_balance_loss(probability, mask, 3)
        loss.backward()
        self.assertAlmostEqual(float(loss), 1., places=6)
        self.assertTrue(torch.allclose(logits.grad, torch.zeros_like(logits), atol=1e-7))

    def test_collapsed_load_pushes_up_the_excluded_expert(self):
        logits = torch.tensor([[.2, .14, .27, .39]]).log().repeat(16, 1).requires_grad_()
        probability = logits.softmax(1)
        mask = torch.zeros_like(probability).scatter_(1, probability.topk(3, 1).indices, 1)
        loss = load_balance_loss(probability, mask, 3)
        loss.backward()
        self.assertGreater(float(loss), 1.)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue((logits.grad[:, 1] < 0).all())
        self.assertTrue((logits.grad[:, [0, 2, 3]] > 0).all())

    def test_k_equals_n_has_constant_baseline(self):
        logits = torch.randn(16, 4, requires_grad=True)
        loss = load_balance_loss(logits.softmax(1), torch.ones_like(logits), 4)
        loss.backward()
        self.assertAlmostEqual(float(loss), 1., places=6)
        self.assertLess(float(logits.grad.abs().max()), 1e-7)

    def test_effective_batch_statistic_matches_definition(self):
        logits = torch.tensor([[4., 0., 0., 0.], [0., 3., 0., 0.],
                               [2., 0., 0., 0.], [3., 0., 0., 0.]], requires_grad=True)
        p = logits.softmax(1)
        mask = torch.zeros_like(p).scatter_(1, p.topk(1, 1).indices, 1)
        full = load_balance_loss(p, mask, 1)
        expected = 4*(mask.detach().mean(0)*p.mean(0)).sum()
        self.assertTrue(torch.allclose(full, expected))
        micro = torch.stack([load_balance_loss(p[i:i+2], mask[i:i+2], 1)
                             for i in (0, 2)]).mean()
        self.assertGreater(float((full-micro).abs()), 1e-3)


if __name__ == '__main__':
    unittest.main()
