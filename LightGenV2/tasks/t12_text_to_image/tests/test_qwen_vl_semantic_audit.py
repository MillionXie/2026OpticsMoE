import unittest
import torch

from LightGenV2.tasks.t12_text_to_image.qwen_vl_semantic_audit import retrieval_scores


class SemanticAuditTests(unittest.TestCase):
    def test_retrieval_ranking_preserves_pair_identity_and_rejects_ties(self):
        text = torch.eye(3)
        rows = retrieval_scores(torch.tensor([[0., 2., 0.], [1., 1., 0.], [1., 0., 0.]]), text, [1, 0, 2])
        self.assertTrue(rows[0]['strict_top1'])
        self.assertEqual(rows[0]['rank'], 1)
        self.assertFalse(rows[1]['strict_top1'])
        self.assertEqual(rows[1]['rank'], 2)
        self.assertFalse(rows[2]['strict_top1'])
        self.assertLess(rows[2]['margin_to_best_alternative'], 0)
