import unittest
import torch

from LightGenV2.tasks.t12_text_to_image.qwen_vl_native_readout import native_layout, valid_native_migration
from LightGenV2.tasks.t12_text_to_image.qwen_vl_postfree import focus_features


class NativeReadoutTests(unittest.TestCase):
    def test_prefix_moves_instruction_start_but_not_suffix_or_original_processor_positions(self):
        layout={'positions':list(range(4,68)),'instruction_bounds':(70,4)}
        migrated=native_layout(layout)
        self.assertEqual(migrated['qwen_image_positions'],layout['positions'])
        self.assertEqual(migrated['instruction_bounds'],(134,4))
        h=torch.randn(1,154,8);valid=torch.ones(1,154,dtype=torch.bool)
        _,mask=focus_features(h,valid,torch.arange(64),*migrated['instruction_bounds'],.15)
        self.assertEqual(torch.where(mask[0])[0].tolist(),list(range(134,150)))
        self.assertEqual(layout['instruction_bounds'],(70,4))
        with self.assertRaises(ValueError):native_layout(migrated)

    def test_layout_migration_requires_explicit_flag_and_verified_native_cache(self):
        cfg={'semantic_prefix':64,'allow_native_readout_migration':True};state={'config':{}}
        self.assertTrue(valid_native_migration(cfg,state,True))
        self.assertFalse(valid_native_migration(cfg,state,False))
        self.assertFalse(valid_native_migration({'semantic_prefix':64},state,True))
