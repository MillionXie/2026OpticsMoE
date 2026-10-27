import unittest

import torch

from ..qwen_vl_hybrid import IdentityBlock, MacroStageBlock
from ..qwen_vl_postfree import apply_postfree, focus_features, instruction_bounds, short_prompt, QuietFusion
from ..modeling import ScaleMatchedFusion


class PostfreeTest(unittest.TestCase):
    def test_quiet_fusion_same_output_and_gradients(self):
        original = ScaleMatchedFusion(.65, .6, .9, 1e-6)
        quiet = QuietFusion(.65, .6, .9, 1e-6)
        quiet.load_state_dict(original.state_dict())
        e = torch.randn(2, 5, 8, requires_grad=True); o = torch.randn(2, 5, 8, requires_grad=True)
        a, b = original(e, o), quiet(e, o)
        self.assertTrue(torch.equal(a, b))
        ga = torch.autograd.grad(a.square().sum(), (e, o, original.raw_alpha))
        gb = torch.autograd.grad(b.square().sum(), (e, o, quiet.raw_alpha))
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(ga, gb)))

    def test_exactly_ten_original_tf_and_original_optical_geometry(self):
        root = torch.nn.Module(); root.model = torch.nn.Module()
        root.model.visual = torch.nn.Module(); root.model.language_model = torch.nn.Module()
        root.model.visual.blocks = torch.nn.ModuleList([torch.nn.Linear(1, 1) for _ in range(24)])
        root.model.language_model.layers = torch.nn.ModuleList([torch.nn.Linear(1, 1) for _ in range(28)])
        macros = apply_postfree(root)
        self.assertEqual(sum(isinstance(b, torch.nn.Linear) for b in root.model.visual.blocks), 6)
        self.assertEqual(sum(isinstance(b, torch.nn.Linear) for b in root.model.language_model.layers), 4)
        self.assertTrue(all(isinstance(b, IdentityBlock) for b in root.model.visual.blocks[18:]))
        self.assertTrue(all(isinstance(b, IdentityBlock) for b in root.model.language_model.layers[24:]))
        self.assertIsInstance(root.model.visual.blocks[6], MacroStageBlock)
        self.assertIsInstance(root.model.language_model.layers[4], MacroStageBlock)
        self.assertEqual(macros['vision'].diagnostics()['physical_active_roi'], 478)
        self.assertEqual(macros['language'].diagnostics()['optical_cycles_per_forward'], 3)
        self.assertTrue(all(float(f.alpha.detach()) >= .6 for m in macros.values() for f in m.fusions))

    def test_instruction_only_mask_and_text_affects_spatial_readout(self):
        h = torch.randn(2, 10, 8); valid = torch.ones(2, 10, dtype=torch.bool)
        valid[1, 9] = False
        visual, mask = focus_features(h, valid, [1, 2], 4, 2, .1)
        self.assertTrue(mask[0, 4:8].all()); self.assertFalse(mask[:, :4].any())
        self.assertFalse(mask[1, 7:].any())
        changed = h.clone(); changed[:, 4:7, 0] += 5
        actual, _ = focus_features(changed, valid, [1, 2], 4, 2, .1)
        self.assertFalse(torch.equal(visual, actual))
        unconditioned, _ = focus_features(h, valid, [1, 2], 4, 2, 0.)
        self.assertTrue(torch.equal(unconditioned, h[:, [1, 2]].float()))
        self.assertEqual(instruction_bounds(torch.tensor([8, 7, 6, 3, 4, 5, 9, 2]), 6, 9), (3, 2))
        with self.assertRaises(ValueError):
            instruction_bounds(torch.tensor([6, 9]), 6, 9)

    def test_short_instruction_preserves_target_attributes_and_scene(self):
        row = {'category': 'lamp', 'mode': 'object', 'target_scene': ['study', 'warm', 'bright', 'left']}
        result = short_prompt(row, 'a red desk lamp with a circular base, entirely visible')
        self.assertIn('red desk lamp with a circular base', result)
        self.assertNotIn('entirely visible', result)
        row['mode'] = 'joint'
        self.assertIn('left window light', short_prompt(row, 'a red desk lamp'))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
