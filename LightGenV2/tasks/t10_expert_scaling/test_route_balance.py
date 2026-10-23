"""CPU checks of load-aware routing gradients and batch semantics."""
import unittest
from pathlib import Path
import tempfile

import torch

from .diagnose_topk import load_balance_loss, classification_margin_loss, selection_key
from .k4_repeat_schedule import ready_job, aggregate_metrics


class RouteBalanceTests(unittest.TestCase):
    def test_repeat_summary_never_reports_single_seed_sd_as_zero(self):
        row = dict(train_accuracy=.8, val_accuracy=.7, test_accuracy=.6)
        self.assertIsNone(aggregate_metrics([row])['test_sample_sd'])
        self.assertIsNone(aggregate_metrics([])['test_mean'])
        self.assertAlmostEqual(aggregate_metrics([row, dict(row, test_accuracy=.8)])['test_sample_sd'],
                               .2/(2**.5))

    def test_repeat_dependency_and_seed_gpu_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'source'
            source.mkdir()
            (source/'result.json').write_text('{}')
            base = dict(source=str(source), stage=1)
            organ17 = dict(base, dataset='organcmnist', seed=17, out='organ17')
            organ27 = dict(base, dataset='organcmnist', seed=27, out='organ27')
            kather27 = dict(base, dataset='kather2016', seed=27, out='kather27')
            jobs = [organ17, organ27, kather27]
            self.assertEqual(ready_job(jobs, {}, 27), organ27)
            running = {(1, 'organcmnist'): (None, None, organ27)}
            self.assertEqual(ready_job(jobs, running, 27), kather27)
            blocked = dict(organ27, source='organ27', stage=2, out='organ27_stage2')
            self.assertIsNone(ready_job([blocked], running, 27))

    def test_accuracy_selection_uses_nll_only_as_tiebreaker(self):
        good_accuracy = dict(accuracy=.75, macro_nll=.9)
        good_nll = dict(accuracy=.72, macro_nll=.8)
        self.assertLess(selection_key(good_accuracy, 'accuracy'), selection_key(good_nll, 'accuracy'))
        self.assertLess(selection_key(good_nll, 'macro_nll'), selection_key(good_accuracy, 'macro_nll'))

    def test_margin_pushes_up_true_class_on_mistake(self):
        logits = torch.tensor([[0., 1., -.5]], requires_grad=True)
        loss = classification_margin_loss(logits.softmax(1), torch.tensor([0]))
        loss.backward()
        self.assertLess(float(logits.grad[0, 0]), 0.)
        self.assertGreater(float(logits.grad[0, 1]), 0.)
        self.assertTrue(torch.isfinite(logits.grad).all())

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
