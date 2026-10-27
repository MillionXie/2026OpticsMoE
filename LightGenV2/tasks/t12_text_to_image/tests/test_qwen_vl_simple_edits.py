import unittest
import numpy as np
from ..qwen_vl_simple_edits import simple_pairs, product_tag, studio_background, PROTOCOL
from ..qwen_vl_turbo_data import add_descriptions, validate_roles
from ..qwen_vl_postfree import short_prompt


class SimpleEditsTests(unittest.TestCase):
    def fixtures(self):
        rows, descriptions = [], {}
        for split in ('train', 'val'):
            for cat, labels in (('lamp', ('white cylindrical shade', 'gray tapered fabric shade')),
                                ('pillow', ('blue and white striped cushion', 'gray geometric cushion'))):
                for i, desc in enumerate(labels):
                    for j in range(2):
                        sid = f'{split}-{cat}-{i}-{j}'
                        p = {'sample_id': sid, 'product_id': sid, 'family_id': sid, 'rgba_sha256': sid, 'category': cat}
                        descriptions[sid] = desc
                        rows.append({'split': split, 'source': p, 'target': p})
        return rows, descriptions

    def test_short_visible_edits_and_coarse_family_is_not_one_sku(self):
        old, descriptions = self.fixtures()
        rows = simple_pairs(old, descriptions, {'train': 2, 'val': 1})
        self.assertTrue(validate_roles(rows)['role_disjoint'])
        for row in rows:
            self.assertLessEqual(len(row['prompt'].split()), 14)
            self.assertEqual(row['edit_count'], 2 if row['mode'] == 'joint' else 1)
            if row['mode'] == 'background':
                self.assertEqual(row['source'], row['target'])
            else:
                self.assertNotEqual(row['source_family'], row['requested_family'])
                self.assertNotEqual(row['source']['product_id'], row['target']['product_id'])
            if row['mode'] == 'object':
                self.assertEqual(row['source_background'], row['target_background'])
            else:
                self.assertNotEqual(row['source_background'], row['target_background'])
        self.assertTrue(all(len({r['target']['product_id'] for r in rows if r['split'] == 'train' and r['mode'] == 'object' and r['category'] == cat and r['requested_family'] == tag}) > 1
                            for cat, tag in (('lamp', 'tapered'), ('pillow', 'striped'))))

    def test_caching_never_expands_short_instructions(self):
        old, descriptions = self.fixtures();rows = simple_pairs(old, descriptions, {'train': 1, 'val': 1})
        self.assertEqual(add_descriptions(rows, descriptions), rows)
        self.assertTrue(all(short_prompt(r, 'irrelevant full product description') == r['prompt'] for r in rows))

    def test_shade_label_does_not_use_cylindrical_base_or_unlabelled_color(self):
        self.assertEqual(product_tag('lamp', 'a cylindrical base with a tapered fabric shade'), 'tapered')
        self.assertIsNone(product_tag('lamp', 'a red metal desk lamp with a curved arm'))
        self.assertIsNone(product_tag('pillow', 'a floral cushion'))

    def test_background_is_deterministic_continuous_studio_field(self):
        first = np.asarray(studio_background('pale_blue'))
        self.assertEqual(first.shape, (256, 256, 3))
        self.assertTrue(np.array_equal(first, np.asarray(studio_background('pale_blue'))))
        self.assertFalse(np.array_equal(first, np.asarray(studio_background('light_gray'))))
        self.assertLess(np.abs(np.diff(first.astype(float), axis=0)).max(), 8)

    def test_no_test_product_can_enter_preparation(self):
        old, descriptions = self.fixtures();old[0]['split'] = 'test'
        with self.assertRaises(ValueError):
            simple_pairs(old, descriptions)


if __name__ == '__main__':
    unittest.main()
