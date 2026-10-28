import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from ..qwen_vl_controlled_room_data import (PROTOCOL, PALETTES, check_reference_target,
    controlled_pairs, literal_prompt_statistics, pixel_edit_mask, render_controlled_pair, room_background)


def synthetic_prior():
    rows = []; descriptions = {}
    for split in ('train', 'val'):
        for category, families in (('lamp', ('cylindrical shade', 'tapered shade')),
                                   ('pillow', ('striped', 'geometric'))):
            for index in range(8):
                identity = f'{split}:{category}:{index}'
                product = {'sample_id': identity, 'product_id': identity, 'family_id': identity,
                    'rgba_sha256': identity, 'category': category, 'image_path': identity, 'mask_path': identity}
                descriptions[identity] = f'a {families[index % 2]} {category}'
                rows.append({'split': split, 'category': category, 'mode': 'background', 'source': product, 'target': product})
    return rows, descriptions


class ControlledRoomDataTests(unittest.TestCase):
    def test_wall_palette_does_not_change_window_or_table_pixels(self):
        first, wall = room_background('light_gray', 64, True)
        second, second_wall = room_background('warm_beige', 64, True)
        a = np.asarray(first); b = np.asarray(second)
        np.testing.assert_array_equal(wall, second_wall)
        np.testing.assert_array_equal(a[~wall], b[~wall])
        self.assertGreater(int(np.any(a != b, axis=-1)[wall].sum()), 500)

    def test_contexts_share_rgb_noise_three_tasks_and_generic_prompts_cover_multiple_targets(self):
        rows, descriptions = synthetic_prior()
        result = controlled_pairs(rows, descriptions, {'seed': 1042,
            'reference_contexts_per_category': {'train': 32, 'val': 16}})
        self.assertEqual(len(result), (32+16)*2*3)
        groups = {}
        contrasting = {}
        target_map = {}
        for row in result:
            self.assertEqual(row['render_protocol'], PROTOCOL)
            self.assertLessEqual(len(row['prompt'].split()), 16)
            value = row['source']['sample_id'], row['source_wall_color'], row['noise_seed']
            self.assertEqual(groups.setdefault(row['reference_context_id'], value), value)
            contrasting.setdefault((row['split'], row['reference_context_id'], row['mode']), []).append(row)
            self.assertNotIn(row['source']['product_id'], row['prompt'])
            if row['mode'] == 'object':
                self.assertEqual(row['source_wall_color'], row['target_wall_color'])
                key = row['split'], row['source']['product_id'], row['requested_family']
                self.assertEqual(target_map.setdefault(key, row['target']['product_id']), row['target']['product_id'])
            else:
                self.assertNotEqual(row['source_wall_color'], row['target_wall_color'])
        for (split, context_id, mode), items in contrasting.items():
            if split == 'train':
                self.assertEqual(len(items), 2)
                self.assertEqual(len({r['prompt'] for r in items}), 2)
                self.assertEqual(len({r['noise_seed'] for r in items}), 1)
                if mode != 'background':self.assertEqual(len({r['target']['product_id'] for r in items}), 2)
            else:self.assertEqual(len(items), 1)
        for record in literal_prompt_statistics(result):
            if record['split'] == 'train' and record['mode'] == 'object':
                self.assertEqual(record['unique_target_products'], 4)
                self.assertLessEqual(record['maximum_target_share'], .3)
                self.assertGreater(record['conditional_target_entropy_bits'], 1.9)

    def test_object_substitution_preserves_background_outside_true_alpha_union(self):
        rows, descriptions = synthetic_prior()
        row = next(r for r in controlled_pairs(rows, descriptions, {'reference_contexts_per_category': {'train': 32, 'val': 16}}) if r['mode'] == 'object')
        def normalized(product, size):
            rgb = np.ones((size, size, 3), dtype=np.uint8)*255
            alpha = np.zeros((size, size), dtype=np.uint8)
            if product['product_id'] == row['source']['product_id']:
                rgb[10:32, 16:28] = (180, 35, 30); alpha[10:32, 16:28] = 255
            else:
                rgb[15:40, 27:38] = (40, 65, 190); alpha[15:40, 27:38] = 255
            return Image.fromarray(rgb), Image.fromarray(alpha)
        with patch('LightGenV2.tasks.t12_text_to_image.qwen_vl_controlled_room_data.normalized_product', normalized):
            before, after = (np.asarray(v) for v in render_controlled_pair(row, 64))
        union = np.zeros((64, 64), dtype=bool); union[10:32, 16:28] = True; union[15:40, 27:38] = True
        np.testing.assert_array_equal(before[~union], after[~union])
        self.assertTrue(np.any(before[union] != after[union]))
        mask = np.asarray(pixel_edit_mask(Image.fromarray(before), Image.fromarray(after))) > 0
        self.assertFalse(mask[~union].any())
        np.testing.assert_array_equal(mask, np.any(before != after, axis=-1))

    def test_noop_supervision_is_rejected(self):
        image = Image.new('RGB', (32, 32), (100, 100, 100))
        with self.assertRaises(ValueError):pixel_edit_mask(image, image.copy())

    def test_ambiguous_identical_inputs_are_not_hidden_with_a_different_seed(self):
        seen = {}; check_reference_target(seen, b'reference', 'instruction', 1, b'target')
        check_reference_target(seen, b'reference', 'instruction', 2, b'target')
        with self.assertRaises(ValueError):check_reference_target(seen, b'reference', 'instruction', 2, b'another target')

    def test_missing_target_diversity_or_test_rows_fail(self):
        rows, descriptions = synthetic_prior()
        reduced = [r for r in rows if not (r['split'] == 'train' and r['category'] == 'pillow' and r['source']['product_id'].endswith((':4', ':6')))]
        with self.assertRaises(ValueError):controlled_pairs(reduced, descriptions,
            {'reference_contexts_per_category': {'train': 24, 'val': 16}})
        rows[0] = {**rows[0], 'split': 'test'}
        with self.assertRaises(ValueError):controlled_pairs(rows, descriptions, {})


if __name__ == '__main__':unittest.main()
