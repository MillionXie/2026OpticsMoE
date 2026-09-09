import unittest
import numpy as np
import torch
from types import SimpleNamespace
from torch import nn
import torch.nn.functional as F
from LightGenV2.tasks.t07_abo_image_retrieval.refinement import (
    CrossProductBatchSampler, lr_multiplier, inflate_convolution, ResidualRetrievalReadout, TrainProductBank,
)
from experiments.qwen3_vl_embedding_2b_caltech101_electronic_retrieval.modeling import ElectronicRetrievalReadout
from LightGenV2.tasks.t07_abo_image_retrieval.retrieval_contract import _ranking_metrics
from LightGenV2.tasks.t07_abo_image_retrieval.run import supcon, category_anchors, TASK
from LightGenV2.tasks.t01_object_retrieval.settings import load_settings


class ContractTests(unittest.TestCase):
    def test_polish_pair_changes_only_phase_rate(self):
        from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.settings import _read_config
        a = _read_config(TASK / "configs/polish_low_lr.yaml")
        b = _read_config(TASK / "configs/polish_phase_reheat.yaml")
        self.assertEqual(a["abo_image_image"], b["abo_image_image"])
        self.assertEqual(a["abo_image_image"]["semantic_anchor_weight"], 1.)
        self.assertEqual(a["abo_image_image"]["teacher_kd_weight"], .1)
        self.assertEqual(a["training"]["epochs"], 30)
        self.assertEqual(b["training"]["phase_learning_rate"], 10*a["training"]["phase_learning_rate"])
        for key in ("learning_rate", "adapter_learning_rate", "readout_learning_rate", "router_learning_rate"):
            self.assertEqual(a["training"][key], b["training"][key])

    def test_train_gallery_excludes_own_product_and_has_gradients(self):
        samples = [SimpleNamespace(product_id=p, category_id=p//2, split="train")
                   for p in range(4) for _ in range(2)]
        features = torch.eye(2).repeat_interleave(4, dim=0)
        bank = TrainProductBank(samples, features, "cpu")
        bank.refresh(features)
        query = torch.tensor([[1., 0.]], requires_grad=True)
        loss, relation = bank.losses(query, [0])
        self.assertLess(float(relation.detach().abs()), 1e-6)
        self.assertGreater(float(bank.losses(query.flip(1), [0])[0].detach()), float(loss.detach()))
        # Corrupt all views of query's own product: excluded column must not matter.
        bank.memory[:2] = torch.tensor([0., 1.])
        torch.testing.assert_close(loss, bank.losses(query, [0])[0])
        (loss + relation).backward()
        self.assertTrue(torch.isfinite(query.grad).all())
        self.assertFalse(bank.memory.requires_grad)
        with self.assertRaises(ValueError):
            bank.update([0, 0], query.detach().repeat(2, 1))
        samples[0].split = "test"
        with self.assertRaises(ValueError):
            TrainProductBank(samples, features, "cpu")

    def test_gallery_profiles_keep_optics_and_no_enhanced_graph(self):
        from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.settings import _read_config
        for name in ("refine_gallery.yaml", "refine_gallery_relation.yaml"):
            path = TASK / "configs" / name
            settings = load_settings(path)
            options = _read_config(path)["abo_image_image"]
            self.assertEqual(settings.top_k, 2)
            self.assertEqual(settings.router_backend, "optical")
            self.assertEqual(settings.embedding_dim, 64)
            self.assertFalse(options.get("enhanced_electronics", False))
            self.assertEqual(len(options["refinement_checkpoint_sha256"]), 64)
            self.assertTrue(options["restore_training_phase_dropout"])

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
