import unittest
from unittest.mock import patch
from pathlib import Path

import torch

from ..qwen_vl_native_prior_data import alignment_specs
from ..qwen_vl_postfree_run import build_generator


class NativePriorTests(unittest.TestCase):
    def test_text_alignment_uses_only_train_and_is_not_new_image_pairs(self):
        rows = []
        for cat in ('lamp', 'pillow'):
            for split in ('val', 'train'):
                for i in range(4):
                    rows.append({'split': split, 'category': cat, 'source': {'product_id': split+cat+str(i)},
                                 'source_scene': ['modern_study', 'warm', 'dim', 'left']})
        specs = alignment_specs(rows)
        self.assertEqual(len(specs), 128)
        self.assertEqual(len({s['source_index'] for s in specs}), 8)
        self.assertTrue(all(rows[s['source_index']]['split'] == 'train' for s in specs))
        self.assertTrue(any('red' in s['prompt'] for s in specs))
        self.assertTrue(any('blue' in s['prompt'] for s in specs))
        self.assertTrue(all('target_image' not in s for s in specs))

    def test_public_prior_never_copies_adapted_cross_attention(self):
        parent = {'config': {'unet_training': 'cross_attention'},
                  'visual_layout': {'positions': [0]}, 'unet_trainable': {'bad_old_cross': torch.ones(1)}}
        stub = torch.nn.Linear(1, 1)
        with patch('LightGenV2.tasks.t12_text_to_image.qwen_vl_postfree_run.build_spatial_head', return_value=stub) as builder:
            build_generator(Path('unused'), parent, {}, torch.device('cpu'), compressed=False, public_prior=True)
            passed = builder.call_args.args[1]
            self.assertEqual(passed['unet_trainable'], {})
            self.assertEqual(passed['config']['unet_training'], 'lora')
            self.assertIn('bad_old_cross', parent['unet_trainable'])
        with self.assertRaises(ValueError):
            build_generator(Path('unused'), parent, {}, torch.device('cpu'), compressed=True, public_prior=True)


if __name__ == '__main__':
    unittest.main()
