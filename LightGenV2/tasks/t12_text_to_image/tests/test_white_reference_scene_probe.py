import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from ..white_reference_scene_probe import conditions, choose_sources, checked_tokens, digest, native_caption


class ProbeContractTests(unittest.TestCase):
    def rows(self):
        return [{'category': c, 'split': 'train', 'source': {'category': c, 'product_id': str(i)}}
                for i, c in enumerate(['lamp', 'lamp', 'pillow', 'pillow'])]

    def test_train_only(self):
        rows = self.rows(); self.assertEqual(len(choose_sources(rows, [0, 1, 2, 3])), 4)
        rows[2]['split'] = 'test'
        with self.assertRaises(ValueError): choose_sources(rows, [0, 1, 2, 3])

    def test_duplicate_products_rejected(self):
        rows = self.rows(); rows[1]['source']['product_id'] = '0'
        with self.assertRaises(ValueError): choose_sources(rows, [0, 1, 2, 3])

    def test_complex_scene_attributes(self):
        for category in ['lamp', 'pillow']:
            for row in conditions(category):
                self.assertTrue(row['prompt'].startswith('Generate'))
                self.assertGreaterEqual(len(row['requested_attributes']), 6)
                self.assertIn('light_side', row['requested_attributes'])
                self.assertGreaterEqual(len(row['requested_attributes']['background_elements']), 3)

    def test_identical_category_prompts_across_products(self):
        self.assertEqual(conditions('lamp'), conditions('lamp'))
        self.assertNotIn('B073', str(conditions('lamp')))

    def test_full_prompt_overflow_rejected(self):
        class Tokenizer:
            model_max_length = 3
            def __call__(self, text, **kwargs):
                self.kwargs = kwargs
                return {'input_ids': list(range(len(text.split())))}
        t = Tokenizer(); self.assertEqual(checked_tokens(t, 'one two')['full_token_count'], 2)
        self.assertFalse(t.kwargs['truncation'])
        with self.assertRaises(ValueError): checked_tokens(t, 'one two three four')

    def test_artifact_hash_changes(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp)/'input.png'; p.write_bytes(b'original'); before = digest(p)
            p.write_bytes(b'changed'); self.assertNotEqual(before, digest(p))

    def test_explicit_caption_preserves_requested_binding(self):
        lamp = conditions('lamp')[0]; cushion = conditions('pillow')[1]
        self.assertEqual(native_caption('lamp', lamp, 'original'), lamp['prompt'])
        self.assertIn('deep blue fabric lampshade', native_caption('lamp', lamp, 'explicit_caption'))
        self.assertIn('terracotta cushion with broad cream stripes', native_caption('pillow', cushion, 'explicit_caption'))
        with self.assertRaises(ValueError): native_caption('lamp', lamp, 'invented')


if __name__ == '__main__': unittest.main()
