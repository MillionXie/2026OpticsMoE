import unittest

import torch

from LightGenV2.tasks.t12_text_to_image.qwen_vl_hybrid import OpticalMacro, TokenCodec, IdentityBlock, MacroStageBlock, apply_hybrid, feature_loss, hybrid_config, load_macro_state, optimizer_groups


class HybridTests(unittest.TestCase):
    def test_native_visual_field_layout_and_overflow(self):
        codec = TokenCodec('vision', 16)
        hidden = torch.randn(1, 256, 16)
        packed = codec.pack(hidden)
        self.assertEqual(tuple(packed.shape), (1, 224, 224))
        self.assertTrue(torch.equal(packed.reshape(1, 256, 196), codec.input(codec.norm(hidden))))
        self.assertEqual(tuple(codec.unpack(packed, 256).shape), hidden.shape)
        with self.assertRaises(ValueError):
            codec.pack(hidden[:, :224])
        with self.assertRaises(ValueError):
            TokenCodec('language', 16).pack(torch.zeros(1, 225, 16))

    def test_two_pairs_have_one_router_real_physics_and_backward(self):
        torch.manual_seed(42)
        macro = OpticalMacro('language', 16, pairs=2, residual_width=8).train()
        x = torch.randn(1, 7, 16)
        y = macro(x)
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertEqual(macro.router_calls, 1)
        self.assertEqual(len(macro.last_stages), 4)
        self.assertIsInstance(macro.paths[1].core.router, torch.nn.Identity)
        feature_loss(y, x*.8+.2).backward()
        phases = [(n, p) for n, p in macro.named_parameters() if 'raw_phase' in n or 'raw_router_phase' in n]
        self.assertTrue(phases)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in phases))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in macro.parameters() if p.grad is not None))
        for f in macro.fusions:
            self.assertGreaterEqual(float(f.alpha.detach()), .60)
            with torch.no_grad():
                f.raw_alpha.fill_(-100)
            self.assertGreaterEqual(float(f.alpha.detach()), .60)
        self.assertEqual(macro.diagnostics()['optical_cycles_per_forward'], 5)
        self.assertEqual(hybrid_config(2, 1)['alpha_min'], .60)

    def test_native_spans_removed_and_deepstack_taps_preserved(self):
        root = torch.nn.Module(); root.model = torch.nn.Module()
        root.model.visual = torch.nn.Module()
        root.model.language_model = torch.nn.Module()
        root.model.visual.blocks = torch.nn.ModuleList([torch.nn.Linear(1, 1) for _ in range(24)])
        root.model.language_model.layers = torch.nn.ModuleList([torch.nn.Linear(1, 1) for _ in range(28)])
        original = root.model.visual.blocks[5]
        macros = apply_hybrid(root)
        self.assertIs(root.model.visual.blocks[5], original)
        self.assertEqual(sum(isinstance(x, torch.nn.Linear) for x in root.model.visual.blocks), 12)
        self.assertEqual(sum(isinstance(x, torch.nn.Linear) for x in root.model.language_model.layers), 8)
        self.assertIsInstance(root.model.visual.blocks[6], MacroStageBlock)
        self.assertIsInstance(root.model.visual.blocks[11], IdentityBlock)
        self.assertIsInstance(root.model.visual.blocks[12], MacroStageBlock)
        self.assertIsInstance(root.model.visual.blocks[17], IdentityBlock)
        self.assertTrue(any(p.requires_grad for p in macros.parameters()))
        self.assertFalse(any(p.requires_grad for p in original.parameters()))

    def test_resume_compatibility_and_phase_learning_rate_partition(self):
        macros = torch.nn.ModuleDict({'language': OpticalMacro('language', 16, residual_width=8)})
        old = {k: v for k, v in macros.state_dict().items() if not k.endswith('.routing_load')}
        report = load_macro_state(macros, old)
        self.assertEqual(report['initialized_training_only_load_ema'], ['language.routing_load'])
        groups = optimizer_groups(macros, {'joint_lr': 1e-5, 'phase_lr': 3e-4, 'router_lr': 5e-4})
        self.assertEqual({g['name']: g['lr'] for g in groups}, {'electronics': 1e-5, 'phase': 3e-4, 'router_phase': 5e-4})
        params = [id(p) for g in groups for p in g['params']]
        self.assertEqual(len(params), len(set(params)))
        self.assertEqual(set(params), {id(p) for p in macros.parameters()})
        del old[next(k for k in old if 'raw_router_phase' in k)]
        with self.assertRaises(ValueError):
            load_macro_state(macros, old)

    def test_router_batch_training_only_and_variable_prompt_lengths(self):
        from LightGenV2.tasks.t12_text_to_image.qwen_vl_hybrid_run import router_batch_balance
        macros = torch.nn.ModuleDict({'language': OpticalMacro('language', 16, residual_width=8)})
        features = [{'split': 'train', 'language_input': torch.randn(1, n, 16)} for n in (3, 5, 7)]
        loss = router_batch_balance(macros, features, [0, 1, 2], torch.device('cpu'))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradient = macros['language'].paths[0].core.router.raw_router_phase.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0)
        features[1]['split'] = 'val'
        with self.assertRaises(ValueError):
            router_batch_balance(macros, features, [0, 1], torch.device('cpu'))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
