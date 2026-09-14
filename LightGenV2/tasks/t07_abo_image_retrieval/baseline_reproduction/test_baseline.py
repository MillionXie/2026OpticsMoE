import ast
import tempfile
import unittest
from pathlib import Path
import torch

try:
    from . import baseline as b
except ImportError:
    import baseline as b


class BaselineTests(unittest.TestCase):
    def test_same_category_is_not_same_sku(self):
        rows = [dict(sample_id='a1', product_id='a', category_id=0, split='gallery'),
                dict(sample_id='b1', product_id='b', category_id=0, split='gallery'),
                dict(sample_id='a2', product_id='a', category_id=0, split='query')]
        z = torch.tensor([[1., 0.], [0., 1.], [0., 1.]])
        result, predictions = b.evaluate_vectors(z, rows, 'enrolled_sku')
        self.assertEqual(result['hit_at_1'], 0)
        self.assertEqual(result['same_ranking_category_hit_at_1_diagnostic'], 1)
        self.assertEqual(result['top1_same_category_wrong_sku_count'], 1)
        self.assertEqual(predictions[0]['top1_product_id'], 'b')

    def test_legacy_uses_product_centroids_and_category_relevance(self):
        rows = [dict(sample_id=str(i), product_id=p, category_id=c, split=s) for i, (p, c, s) in
                enumerate([('a', 0, 'gallery'), ('a', 0, 'gallery'), ('b', 1, 'gallery'), ('q', 0, 'query')])]
        result, _ = b.evaluate_vectors(torch.tensor([[1., 0.], [1., 0.], [0., 1.], [1., 0.]]), rows, 'legacy_category')
        self.assertEqual(result['candidate_count'], 2)
        self.assertEqual(result['hit_at_1'], 1)
        self.assertEqual(result['top1_same_category_wrong_sku_count'], 1)

    def test_legacy_rejects_same_sku_in_query_and_gallery(self):
        rows = [dict(sample_id=str(i), product_id='a', category_id=0, split=s) for i, s in enumerate(['gallery', 'query'])]
        with self.assertRaises(ValueError): b.evaluate_vectors(torch.ones(2, 4), rows, 'legacy_category')

    def test_enrolled_requires_positive(self):
        rows = [dict(sample_id=str(i), product_id=str(i), category_id=0, split=s) for i, s in enumerate(['gallery', 'query'])]
        with self.assertRaises(ValueError): b.evaluate_vectors(torch.ones(2, 4), rows, 'enrolled_sku')

    def test_pooling_left_and_right_padding(self):
        hidden = torch.arange(2*4*3).reshape(2, 4, 3)
        pooled = b.pool_last(hidden, torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0]]))
        torch.testing.assert_close(pooled, torch.stack([hidden[0, 3], hidden[1, 1]]).float())
        with self.assertRaises(ValueError): b.pool_last(hidden, torch.zeros(2, 4))

    def test_metric_hit_is_not_set_recall(self):
        m = b.metrics([[1, 1, 0], [0, 1, 1]])
        self.assertEqual(m['hit_at_1'], .5)
        self.assertEqual(m['positive_recall_at_1'], .25)

    def test_no_invalid_vectors(self):
        for x in [torch.zeros(2, 3), torch.full((2, 3), float('nan'))]:
            with self.assertRaises(ValueError): b.normalize(x)

    def test_sha_constants_and_cache_mismatch(self):
        self.assertEqual(len(b.PARENT_SHA), 64)
        self.assertEqual(len(b.ENROLLED_SHA), 64)
        self.assertTrue(all(len(v) == 64 for v in b.MODEL_FILES_SHA.values()))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'cache.pt'; torch.save({}, path)
            with self.assertRaises(ValueError): b.load_cache(path, [], 'enrolled_sku', 'native', '0'*64)

    def test_duplicate_cache_ids_rejected(self):
        rows = [dict(sample_id='a'), dict(sample_id='b')]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'cache.pt'
            torch.save(dict(manifest_sha256=b.ENROLLED_SHA, ids=['a','a'], vectors=torch.ones(2,64)), path)
            with self.assertRaises(ValueError): b.load_cache(path, rows, 'enrolled_sku', 'native', b.sha256(path))

    def test_no_training_or_parent_project_imports(self):
        tree = ast.parse(Path(b.__file__).read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                self.assertFalse(node.level)
                self.assertFalse((node.module or '').startswith(('LightGenV2', 'experiments')))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, ['backward', 'step'])


if __name__ == '__main__': unittest.main()
