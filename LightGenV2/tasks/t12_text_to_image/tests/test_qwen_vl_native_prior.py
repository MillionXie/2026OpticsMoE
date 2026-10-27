import unittest
from unittest.mock import patch
from pathlib import Path

import torch

from ..qwen_vl_native_prior_data import alignment_specs
from ..qwen_vl_postfree_run import build_generator, validate_rendered_teacher, rendered_teacher_loss
import random


class NativePriorTests(unittest.TestCase):
    def test_rendered_supervision_rejects_non_train_and_wrong_teacher(self):
        aug = {'TRAIN_only': True, 'source_frontend': 'full_frozen_qwen', 'synthetic_target_images': True,
               'teacher_reference_input': False, 'teacher_is_public_native_text_only': True, 'noise_seed': 1042,
               'specs': [{'source_split': 'train'}], 'teacher_latents': [torch.zeros(4, 32, 32)],
               'hidden': [torch.ones(8, 4)]}
        validate_rendered_teacher(aug)
        for changed in ({'TRAIN_only': False}, {'noise_seed': 5}, {'teacher_reference_input': True},
                        {'specs': [{'source_split': 'val'}]}, {'source_frontend': 'postfree'}):
            with self.assertRaises(ValueError):
                validate_rendered_teacher({**aug, **changed})

    def test_prior_loss_reaches_adapter_and_explicitly_disables_spatial(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.adapter = torch.nn.Linear(4, 1)
            def forward(self, hidden, valid, noise, disable_spatial=False, decode=True):
                if not disable_spatial or decode:
                    raise AssertionError('Auxiliary prior loss must disable spatial and RGB decoding')
                z = self.adapter(hidden.mean(1))[:, :, None, None].expand(-1, 4, 32, 32)
                return z, None
        model = Tiny()
        aug = {'hidden': [torch.ones(8, 4)], 'teacher_latents': [torch.ones(4, 32, 32)], 'noise_seed': 1042}
        loss = rendered_teacher_loss(model, aug, random.Random(1), 2, torch.device('cpu'))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(model.adapter.weight.grad).all())

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
