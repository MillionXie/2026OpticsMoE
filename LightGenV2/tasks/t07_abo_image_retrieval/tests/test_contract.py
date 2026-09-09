import unittest
import numpy as np
import torch
from LightGenV2.tasks.t07_abo_image_retrieval.retrieval_contract import _ranking_metrics
from LightGenV2.tasks.t07_abo_image_retrieval.run import supcon, category_anchors, TASK
from LightGenV2.tasks.t01_object_retrieval.settings import load_settings


class ContractTests(unittest.TestCase):
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

    def test_train_only_anchors(self):
        vectors = torch.eye(10).repeat_interleave(2, 0).requires_grad_(True)
        anchors = category_anchors(vectors, torch.arange(10).repeat_interleave(2))
        self.assertTrue(torch.equal(anchors, torch.eye(10)))
        self.assertFalse(anchors.requires_grad)
        with self.assertRaises(ValueError):
            category_anchors(vectors[:4], [0, 0, 1, 1])


if __name__ == '__main__':
    unittest.main()
