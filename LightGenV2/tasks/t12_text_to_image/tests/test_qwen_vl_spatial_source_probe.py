"""CPU contracts for a spatial-only native-Qwen-feature diagnostic override."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from ..qwen_vl_postfree import focus_features
from ..qwen_vl_spatial import SpatialBridge, SpatialInputConv, SpatialTurboHead


class RecordingAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(16, 8)
        self.last_mask = None

    def forward(self, hidden, valid):
        self.last_mask = valid.detach().clone()
        context = (hidden.float()*valid[:, :, None]).sum(1)/valid.sum(1, keepdim=True)
        return self.project(context)[:, None].expand(-1, 3, -1)


class TinyUNet(nn.Module):
    dtype = torch.float32

    def __init__(self):
        super().__init__()
        self.conv_in = SpatialInputConv(nn.Conv2d(4, 4, 3, padding=1), 4)
        with torch.no_grad():
            self.conv_in.spatial.weight.normal_(std=.08)

    def forward(self, value, timestep, encoder_hidden_states):
        text = encoder_hidden_states.mean((1, 2))[:, None, None, None]
        return SimpleNamespace(sample=self.conv_in(value)+.03*text)


class TinyScheduler:
    init_noise_sigma = 1.

    def set_timesteps(self, count, device):
        self.timesteps = torch.tensor([1], device=device)

    def scale_model_input(self, value, timestep):
        return value

    def step(self, prediction, timestep, latents):
        return SimpleNamespace(prev_sample=latents-.1*prediction)


class TinyVAE(nn.Module):
    dtype = torch.float32
    config = SimpleNamespace(scaling_factor=1.)

    def encode(self, *args, **kwargs):
        raise AssertionError('A spatial-feature probe cannot introduce a reference-VAE inference branch')

    def decode(self, value):
        return SimpleNamespace(sample=value[:, :3])


def tiny_head(conditional=True):
    bridge = SpatialBridge(input_dim=16, width=16, channels=4, grid=2, conditional=conditional)
    if conditional:
        with torch.no_grad():
            bridge.film.weight.normal_(std=.03)
            bridge.film.bias.fill_(.01)
    model = SpatialTurboHead(TinyUNet(), TinyVAE(), TinyScheduler(), RecordingAdapter(),
        bridge, [1, 2, 3, 4]).eval()
    model.instruction_span = (6, 2)
    model.text_readout_gain = .15
    model.multimodal_semantic_readout = True
    return model


def fixture():
    hidden = torch.randn(2, 12, 16)
    valid = torch.ones(2, 12, dtype=torch.bool)
    valid[1, 10:] = False
    noise = torch.randn(2, 4, 8, 8)
    return hidden, valid, noise


class SpatialSourceProbeTests(unittest.TestCase):
    def test_probe_batch_orders_reproduce_the_shared400_warm300_paired_budget(self):
        from ..qwen_vl_spatial_source_probe import batch_orders
        cfg = {'seed': 1042, 'warm_steps': 400, 'warm_batch_size': 8,
               'paired_steps': 300, 'batch_size': 4}
        first = batch_orders(60, cfg); second = batch_orders(60, cfg)
        self.assertEqual(first, second)
        self.assertEqual((len(first[0]), len(first[1])), (400, 300))
        for batches, width in zip(first, (8, 4)):
            self.assertTrue(all(len(batch) == width for batch in batches))
            self.assertTrue(all(len(set(batch)) == width for batch in batches))
            self.assertEqual({i for batch in batches for i in batch}, set(range(60)))
        self.assertNotEqual(first, batch_orders(60, {**cfg, 'seed': 1043}))
        with self.assertRaises(ValueError):batch_orders(7, cfg)

    def test_actual_raw_capture_requires_one_finite_native_bf16_merger_call_and_removes_hook(self):
        from ..qwen_vl_spatial_source_probe import capture_raw
        merger = nn.Identity()
        qwen = SimpleNamespace(model=SimpleNamespace(visual=SimpleNamespace(merger=merger)))
        processor = object(); inputs = {}; layout = {}; device = torch.device('cpu')
        hidden = torch.randn(11, 2048, dtype=torch.bfloat16, requires_grad=True)
        native = torch.randn(64, 2048, dtype=torch.bfloat16, requires_grad=True)

        def forward(*args):
            merger(native)
            return hidden

        target = 'LightGenV2.tasks.t12_text_to_image.qwen_vl_spatial_source_probe.raw_hidden'
        with patch(target, side_effect=forward) as mocked:
            fused, captured, calls = capture_raw(qwen, processor, inputs, 0, layout, device)
            mocked.assert_called_once_with(qwen, processor, inputs, 0, layout, device)
        self.assertEqual(calls, 1)
        self.assertEqual((captured.shape, captured.dtype, captured.device.type),
                         (torch.Size([64, 2048]), torch.bfloat16, 'cpu'))
        self.assertTrue(captured.is_contiguous())
        self.assertFalse(captured.requires_grad); self.assertFalse(fused.requires_grad)
        torch.testing.assert_close(captured, native.detach(), rtol=0, atol=0)
        torch.testing.assert_close(fused, hidden.detach(), rtol=0, atol=0)
        self.assertFalse(merger._forward_hooks)
        for call_count in (0, 2):
            def wrong_calls(*args):
                for _ in range(call_count):merger(native)
                return hidden
            with self.subTest(calls=call_count), patch(target, side_effect=wrong_calls):
                with self.assertRaises(ValueError):capture_raw(qwen, processor, inputs, 0, layout, device)
            self.assertFalse(merger._forward_hooks)
        for bad in (native.float(), native[:63], torch.full_like(native, float('nan'))):
            def wrong_field(*args):
                merger(bad)
                return hidden
            with self.subTest(shape=tuple(bad.shape), dtype=bad.dtype), patch(target, side_effect=wrong_field):
                with self.assertRaises(ValueError):capture_raw(qwen, processor, inputs, 0, layout, device)
            self.assertFalse(merger._forward_hooks)

    def test_complete_train_counterfactual_contract_rejects_seed_or_target_corruption(self):
        from ..qwen_vl_spatial_source_probe import counterfactual_contract
        rows = []; references = []; targets = []; manifest = []
        for context in range(10):
            context_id = 'train-context-'+str(context)
            reference = torch.full((3, 4, 4), 10+context, dtype=torch.uint8)
            for mode_index, mode in enumerate(('background', 'object', 'joint')):
                ids = []; prompts = []
                for branch in range(2):
                    sample_id = context_id+':'+mode+':'+str(branch)
                    prompt = mode+' instruction '+str(branch)
                    rows.append({'sample_id': sample_id, 'split': 'train',
                        'category': 'lamp' if context < 5 else 'pillow', 'mode': mode,
                        'reference_context_id': context_id, 'prompt': prompt, 'noise_seed': 1042+context})
                    references.append(reference)
                    targets.append(reference+1+2*mode_index+branch)
                    ids.append(sample_id); prompts.append(prompt)
                manifest.append({'reference_context_id': context_id, 'mode': mode,
                                 'sample_ids': ids, 'prompts': prompts})
        data = {'rows': rows, 'reference': torch.stack(references), 'target': torch.stack(targets)}
        contract = counterfactual_contract(data, manifest)
        self.assertEqual((contract['pairs'], contract['rows'], contract['reference_contexts']), (30, 60, 10))
        self.assertTrue(contract['same_context_RGB_and_noise'])
        self.assertTrue(contract['distinct_paired_prompts_and_targets'])
        bad_seed = {**data, 'rows': [dict(row) for row in rows]}
        bad_seed['rows'][1]['noise_seed'] += 1
        with self.assertRaises(ValueError):counterfactual_contract(bad_seed, manifest)
        # This pair is internally consistent but disagrees with other tasks
        # for the same reference context, which must also be rejected.
        bad_context_seed = {**data, 'rows': [dict(row) for row in rows]}
        for index in (2, 3):bad_context_seed['rows'][index]['noise_seed'] += 1
        with self.assertRaises(ValueError):counterfactual_contract(bad_context_seed, manifest)
        duplicate_target = {**data, 'target': data['target'].clone()}
        duplicate_target['target'][1].copy_(duplicate_target['target'][0])
        with self.assertRaises(ValueError):counterfactual_contract(duplicate_target, manifest)

    def test_original_lm_visual_override_preserves_full_latent_rgb_and_gradients_exactly(self):
        torch.manual_seed(24)
        model = tiny_head(); base_hidden, valid, noise = fixture()
        trainable = tuple(value for value in model.parameters() if value.requires_grad)
        default_hidden = base_hidden.clone().requires_grad_(True)
        default = model(default_hidden, valid, noise)
        default_loss = sum(value.square().mean() for value in default)
        default_gradients = torch.autograd.grad(default_loss, (default_hidden,)+trainable, allow_unused=True)
        replacement_hidden = base_hidden.clone().requires_grad_(True)
        replacement = model(replacement_hidden, valid, noise,
            native_visual=replacement_hidden[:, model.visual_positions])
        replacement_loss = sum(value.square().mean() for value in replacement)
        replacement_gradients = torch.autograd.grad(replacement_loss,
            (replacement_hidden,)+trainable, allow_unused=True)
        for expected, actual in zip(default, replacement):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for expected, actual in zip(default_gradients, replacement_gradients):
            if expected is None:
                self.assertIsNone(actual)
            else:
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_new_native_source_changes_spatial_output_and_gets_gradients_without_changing_semantics(self):
        torch.manual_seed(25)
        model = tiny_head(); hidden, valid, noise = fixture()
        untouched_hidden = hidden.clone()
        original_condition = model.semantic_conditions(hidden, valid).detach().clone()
        original_mask = model.adapter.last_mask.clone()
        original_spatial, _ = model.spatial_inputs(hidden, valid)
        native = torch.randn_like(hidden[:, model.visual_positions]).requires_grad_(True)
        new_spatial, _ = model.spatial_inputs(hidden, valid, native_visual=native)
        self.assertFalse(torch.equal(original_spatial, new_spatial))
        default_output = model(hidden, valid, noise)
        new_output = model(hidden, valid, noise, native_visual=native)
        self.assertFalse(torch.equal(default_output[0], new_output[0]))
        torch.testing.assert_close(model.semantic_conditions(hidden, valid), original_condition, rtol=0, atol=0)
        torch.testing.assert_close(model.adapter.last_mask, original_mask, rtol=0, atol=0)
        torch.testing.assert_close(hidden, untouched_hidden, rtol=0, atol=0)
        sum(value.square().mean() for value in new_output).backward()
        self.assertIsNotNone(native.grad)
        self.assertTrue(torch.isfinite(native.grad).all())
        self.assertGreater(float(native.grad.abs().sum()), 0.)

    def test_malformed_batch_grid_width_or_nonfinite_native_features_are_rejected(self):
        torch.manual_seed(26)
        model = tiny_head(); hidden, valid, _ = fixture()
        bad = [torch.randn(1, 4, 16), torch.randn(2, 3, 16), torch.randn(2, 4, 15),
            torch.randn(4, 16), torch.full((2, 4, 16), float('nan')),
            torch.full((2, 4, 16), float('inf'))]
        for value in bad:
            with self.subTest(shape=tuple(value.shape), finite=bool(torch.isfinite(value).all())):
                with self.assertRaises(ValueError):model.spatial_inputs(hidden, valid, native_visual=value)
                with self.assertRaises(ValueError):
                    focus_features(hidden, valid, model.visual_positions, 6, 2, .15, visual_override=value)

    def test_conditional_film_keeps_original_instruction_context_and_auxiliary_identity(self):
        torch.manual_seed(27)
        model = tiny_head(conditional=True); hidden, valid, noise = fixture()
        old_visual, old_context = model.spatial_inputs(hidden, valid)
        native = torch.randn_like(hidden[:, model.visual_positions])
        new_visual, new_context = model.spatial_inputs(hidden, valid, native_visual=native)
        self.assertFalse(torch.equal(new_visual, old_visual))
        torch.testing.assert_close(new_context, old_context, rtol=0, atol=0)
        default = model.forward_with_aux(hidden, valid, noise, with_aux=True)
        replacement = model.forward_with_aux(hidden, valid, noise, with_aux=True,
            native_visual=hidden[:, model.visual_positions])
        for expected, actual in zip(default, replacement):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_focused_default_matches_historical_formula_and_zero_gain_preserves_source(self):
        torch.manual_seed(28)
        for dtype in (torch.float32, torch.bfloat16):
            hidden, valid, _ = fixture(); hidden = hidden.to(dtype)
            positions = torch.tensor([1, 2, 3, 4])
            instruction_mask = torch.zeros_like(valid)
            instruction_mask[0, 6:10] = True; instruction_mask[1, 6:8] = True
            visual = hidden[:, positions].float()
            context = (hidden.float()*instruction_mask[:, :, None]).sum(1)/instruction_mask.sum(1, keepdim=True)
            vrms = visual.square().mean((1, 2), keepdim=True).sqrt().detach().clamp_min(1e-6)
            crms = context.square().mean(1, keepdim=True).sqrt().detach().clamp_min(1e-6)
            expected = visual+.15*(context/crms)[:, None]*vrms
            actual, mask = focus_features(hidden, valid, positions, 6, 2, .15)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(mask, instruction_mask, rtol=0, atol=0)
            native = torch.randn_like(visual)
            unconditioned, native_mask = focus_features(hidden, valid, positions, 6, 2, 0., visual_override=native)
            torch.testing.assert_close(unconditioned, native, rtol=0, atol=0)
            torch.testing.assert_close(native_mask, instruction_mask, rtol=0, atol=0)


if __name__ == '__main__':unittest.main()
