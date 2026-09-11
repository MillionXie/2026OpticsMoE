import unittest
from types import SimpleNamespace
import torch
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.readout_distill import (
    ridge_fit, training_product_partition, select_ridge, replace_projection,
)


class ReadoutDistillTests(unittest.TestCase):
    def test_ridge_affine_recovery_and_scale(self):
        g = torch.Generator().manual_seed(7)
        x = torch.randn(80, 6, generator=g)
        w = torch.randn(3, 6, generator=g)
        b = torch.randn(3, generator=g)
        y = x@w.T+b
        fitted, intercept, _ = ridge_fit(x, y, 1e-8)
        torch.testing.assert_close(fitted, w, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(intercept, b, atol=1e-6, rtol=1e-6)
        scaled, sb, _ = ridge_fit(10*x, y, 1e-8)
        torch.testing.assert_close(scaled*10, fitted)
        torch.testing.assert_close(sb, intercept)

    def test_reject_invalid_and_constant(self):
        for strength in (0, -1, float('nan')):
            with self.assertRaises(ValueError):ridge_fit(torch.randn(5, 3), torch.randn(5, 2), strength)
        with self.assertRaises(ValueError):ridge_fit(torch.ones(5, 3), torch.randn(5, 2), .1)
        with self.assertRaises(ValueError):ridge_fit(torch.randn(5, 3), torch.randn(4, 2), .1)

    def test_product_disjoint_and_training_only(self):
        samples = [SimpleNamespace(split='train', category_id=c, product_id=f'{c}-{i}')
                   for c in range(2) for i in range(10) for _ in range(3)]
        held = training_product_partition(samples)
        self.assertEqual(int(held.sum()), 12)
        self.assertTrue(torch.equal(held, training_product_partition(samples)))
        a = {s.product_id for s, h in zip(samples, held) if h}
        b = {s.product_id for s, h in zip(samples, held) if not h}
        self.assertFalse(a & b)
        samples[0].split='test'
        with self.assertRaises(ValueError):training_product_partition(samples)

    def test_selection_has_no_test_argument(self):
        g = torch.Generator().manual_seed(3)
        x = torch.randn(60, 8, generator=g)
        y = torch.randn(60, 4, generator=g)
        held = torch.arange(60)%5==0
        w, b, report = select_ridge(x, y, held)
        self.assertEqual(w.shape, (4, 8)); self.assertEqual(b.shape, (4,))
        self.assertEqual(len(report['candidates']), 4)
        winner = max(report['candidates'], key=lambda r:(r['heldout_train_cosine'], r['strength']))
        self.assertEqual(report['selected_strength'], winner['strength'])
        ew, eb, _ = ridge_fit(x, y, winner['strength'])
        torch.testing.assert_close(w, ew); torch.testing.assert_close(b, eb)

    def test_only_projection_changes_no_score_inheritance(self):
        phase = torch.randn(5)
        source = dict(metadata={'retrieval_head':'linear64', 'fusion_alpha_min':.4001},
                      state_dict={'vision.optics.global_phase':phase,
                                  'readout.projection.weight':torch.randn(64,384),
                                  'readout.projection.bias':torch.randn(64)},
                      selection_score=.99, auxiliary_training_head={'stale':True})
        fitted = replace_projection(source, torch.zeros(64,384), torch.zeros(64))
        self.assertIs(fitted['state_dict']['vision.optics.global_phase'], phase)
        self.assertEqual(source['metadata'], fitted['metadata'])
        self.assertNotIn('selection_score', fitted)
        self.assertNotIn('auxiliary_training_head', fitted)
        self.assertFalse(torch.equal(source['state_dict']['readout.projection.weight'], fitted['state_dict']['readout.projection.weight']))
        source['metadata']['retrieval_head']='relu128'
        with self.assertRaises(ValueError):replace_projection(source, torch.zeros(64,384), torch.zeros(64))


if __name__=='__main__':unittest.main()
