import unittest
import numpy as np
import torch
from types import SimpleNamespace
from torch import nn
import torch.nn.functional as F
from LightGenV2.tasks.t07_abo_image_retrieval.refinement import (
    CrossProductBatchSampler, lr_multiplier, inflate_convolution, ResidualRetrievalReadout,
)
from experiments.qwen3_vl_embedding_2b_caltech101_electronic_retrieval.modeling import ElectronicRetrievalReadout
from LightGenV2.tasks.t07_abo_image_retrieval.retrieval_contract import _ranking_metrics
from LightGenV2.tasks.t07_abo_image_retrieval.run import supcon, category_anchors, TASK
from LightGenV2.tasks.t01_object_retrieval.settings import load_settings


class ContractTests(unittest.TestCase):
    def test_cross_product_batch(self):
        samples = [SimpleNamespace(sku_index=c, sku_name=f"{c}-{p}")
                   for c in range(10) for p in range(3) for _ in range(2)]
        sampler = CrossProductBatchSampler(samples, 3, 42)
        batches = list(sampler)
        self.assertEqual(batches, list(sampler))
        for batch in batches:
            self.assertEqual(len(batch), 20)
            for c in range(10):
                self.assertEqual(len({samples[i].sku_name for i in batch if samples[i].sku_index == c}), 2)

    def test_schedule(self):
        self.assertAlmostEqual(lr_multiplier(1, 80), .2)
        self.assertAlmostEqual(lr_multiplier(5, 80), 1.)
        self.assertAlmostEqual(lr_multiplier(80, 80), .1)

    def test_readout_identity_gradient_and_reload(self):
        original = ElectronicRetrievalReadout(384, 64)
        enhanced = ResidualRetrievalReadout(original)
        x = torch.randn(4, 384)
        self.assertTrue(torch.equal(original(x), enhanced(x)))
        enhanced(x).sum().backward()
        self.assertGreater(float(enhanced.correction[-1].weight.grad.abs().sum()), 0)
        restored = ResidualRetrievalReadout(ElectronicRetrievalReadout(384, 64))
        restored.load_state_dict(enhanced.state_dict(), strict=True)
        enhanced.eval(); restored.eval()
        self.assertTrue(torch.equal(enhanced(x), restored(x)))

    def test_kernel_inflation_preserves_spatial_and_causal_outputs(self):
        spatial = nn.Conv2d(4, 4, 5, groups=4, bias=False)
        x = torch.randn(2, 4, 12, 12)
        enlarged = inflate_convolution(spatial, 9, False)
        torch.testing.assert_close(spatial(F.pad(x, (2, 2, 2, 2))), enlarged(F.pad(x, (4, 4, 4, 4))))
        causal = nn.Conv1d(4, 4, 5, groups=4, bias=False)
        x = torch.randn(2, 4, 12)
        enlarged = inflate_convolution(causal, 9, True)
        torch.testing.assert_close(causal(F.pad(x, (4, 0))), enlarged(F.pad(x, (8, 0))))

    def test_hit_is_not_positive_recall(self):
        relevant = np.zeros((2, 120), dtype=bool)
        relevant[:, :12] = True
        values = _ranking_metrics(relevant)
        self.assertEqual(values['hit_at_1'], 1)
        self.assertAlmostEqual(values['positive_recall_at_1'], 1/12)
        self.assertAlmostEqual(values['map_at_10'], 1)

    def test_supcon_learns_category_not_self(self):
        labels = torch.tensor([0, 0, 1, 1])
        aligned = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]], requires_grad=True)
        loss = supcon(aligned, labels)
        self.assertLess(float(loss), float(supcon(aligned[[0, 2, 1, 3]], labels)))
        loss.backward()
        self.assertTrue(torch.isfinite(aligned.grad).all())
        with self.assertRaises(ValueError):
            supcon(aligned, torch.arange(4))

    def test_optical_contract(self):
        settings = load_settings(TASK / 'configs/optical_top2_dc20.yaml')
        self.assertEqual(settings.router_backend, 'optical')
        self.assertEqual(settings.top_k, 2)
        self.assertEqual(settings.embedding_dim, 64)
        self.assertEqual(settings.language_optical_max_shift_pixels, 0)
        self.assertEqual(settings.optical_router_input_shift_pixels, 0)
        self.assertFalse(settings.language_optical_k_space_enabled)
        self.assertEqual(settings.fusion_alpha_initial, .1)
        self.assertFalse(settings.native_pre_attention_enabled)
        self.assertEqual(settings.student_language_mode, "optical_moe")

    def test_train_only_anchors(self):
        vectors = torch.eye(10).repeat_interleave(2, 0).requires_grad_(True)
        anchors = category_anchors(vectors, torch.arange(10).repeat_interleave(2))
        self.assertTrue(torch.equal(anchors, torch.eye(10)))
        self.assertFalse(anchors.requires_grad)
        with self.assertRaises(ValueError):
            category_anchors(vectors[:4], [0, 0, 1, 1])


if __name__ == '__main__':
    unittest.main()
