"""Detector-processing, fused-head shape, and checkpoint identity regressions."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from ..model import GlobalStage, LightGenTemporal, ModelConfig, load_lightgen
from ..tools.simplify_checkpoint import fold_final_projection, prune_hidden_units


class SimplificationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def configuration(self, **changes):
        config = ModelConfig(
            width=8,
            router_hidden=8,
            readout_hidden=8,
            expert_mode='linear',
            orm_mode='amplitude',
            fusion_router_input='all_tokens',
            input_shift=0,
            phase_shift=0,
            ccd_shift=0,
            phase_dropout=0,
            router_noise=0,
        )
        return replace(config, **changes)

    def ready_model(self, **changes):
        torch.manual_seed(121)
        model = LightGenTemporal(self.configuration(**changes)).eval()
        model.set_prompt(torch.randn(38, 2048))
        model.set_quality_levels(torch.tensor([0., 100.]))
        return model

    def test_historical_defaults_preserve_detector_formula(self):
        config = self.configuration()
        self.assertFalse(config.fused_readout)
        self.assertEqual(config.detector_processing, 'legacy_log_clip')
        stage = GlobalStage(config, frame_level=False)
        patches = torch.zeros(2, 3, 10, 10)
        patches[..., 0, 0] = 100.
        relative = patches / patches.mean((-2, -1), keepdim=True).clamp_min(1e-6)
        expected = torch.log1p(relative.clamp_max(8))
        self.assertTrue(torch.equal(stage.process_detector(patches), expected))

    def test_relative_intensity_preserves_values_above_legacy_ceiling(self):
        stage = GlobalStage(
            self.configuration(detector_processing='relative_intensity'),
            frame_level=False,
        )
        patches = torch.zeros(1, 1, 10, 10)
        patches[..., 0, 0] = 100.
        processed = stage.process_detector(patches)
        self.assertEqual(float(processed[0, 0, 0, 0]), 100.)
        self.assertEqual(float(processed.mean()), 1.)
        self.assertTrue(torch.equal(processed, patches))

    def test_relative_intensity_zero_field_has_finite_values_and_gradients(self):
        stage = GlobalStage(
            self.configuration(detector_processing='relative_intensity'),
            frame_level=False,
        )
        patches = torch.zeros(2, 3, 10, 10, requires_grad=True)
        processed = stage.process_detector(patches)
        self.assertTrue(torch.equal(processed, torch.zeros_like(processed)))
        processed.sum().backward()
        self.assertTrue(bool(torch.isfinite(patches.grad).all()))

    def test_fused_joint_global_outputs_96_channels_for_4032_input_head(self):
        config = self.configuration(
            fused_readout=True,
            detector_processing='relative_intensity',
        )
        model = LightGenTemporal(config).eval()
        self.assertIsInstance(model.video_global.output, nn.Identity)
        self.assertIsInstance(model.frame_global.output, nn.Linear)
        self.assertEqual(model.readout[1].in_features, 42 * 96)
        with torch.no_grad():
            output, guard = model.video_global(torch.randn(1, 16, 42, 8))
        self.assertEqual(tuple(output.shape), (1, 16, 42, 96))
        self.assertEqual(tuple(guard.shape), ())
        self.assertTrue(bool(torch.isfinite(output).all()))

    def test_fused_checkpoint_roundtrip_preserves_full_forward(self):
        model = self.ready_model(
            fused_readout=True,
            detector_processing='relative_intensity',
        )
        value = torch.randn(1, 16, 4, 49, 1024)
        with torch.no_grad():
            expected = model(value)['prediction']
        checkpoint = model.checkpoint(fixture='simplification')
        self.assertEqual(checkpoint['schema'], 8)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'fused.pt'
            torch.save(checkpoint, path)
            restored, metadata = load_lightgen(path)
        self.assertEqual(metadata, {'fixture': 'simplification'})
        self.assertEqual(restored.cfg, model.cfg)
        with torch.no_grad():
            actual = restored(value)['prediction']
        self.assertTrue(torch.equal(actual, expected))

    def test_final_projection_folding_preserves_full_model_predictions(self):
        source = self.ready_model()
        value = torch.randn(1, 16, 4, 49, 1024)
        folded = fold_final_projection(source)
        self.assertFalse(source.cfg.fused_readout)
        self.assertTrue(folded.cfg.fused_readout)
        self.assertIsInstance(folded.video_global.output, nn.Identity)
        self.assertEqual(folded.readout[1].in_features, 42 * 96)
        with torch.no_grad():
            expected = source(value)
            actual = folded(value)
        self.assertTrue(torch.allclose(
            actual['quality_logits'], expected['quality_logits'], atol=1e-4, rtol=1e-5))
        self.assertTrue(torch.allclose(
            actual['prediction'], expected['prediction'], atol=1e-4, rtol=1e-5))

    def test_hidden_unit_pruning_copies_selected_relu_basis_and_bias_correction(self):
        source = fold_final_projection(self.ready_model())
        features = torch.randn(24, 42, 96)
        with torch.no_grad():
            activations = source.readout[2](source.readout[1](features.flatten(-2)))
            importance = (activations.square().mean(0).sqrt()
                          * source.readout[3].weight.square().sum(0).sqrt())
            expected_indices = importance.topk(4).indices.sort().values
        pruned, selected = prune_hidden_units(source, features, hidden_width=4)
        self.assertEqual(selected, expected_indices.tolist())
        self.assertEqual(selected, sorted(set(selected)))
        self.assertEqual(len(selected), 4)
        self.assertEqual(source.cfg.readout_hidden, 8)
        self.assertEqual(pruned.cfg.readout_hidden, 4)
        self.assertTrue(torch.equal(
            pruned.readout[1].weight, source.readout[1].weight[expected_indices]))
        self.assertTrue(torch.equal(
            pruned.readout[1].bias, source.readout[1].bias[expected_indices]))
        self.assertTrue(torch.equal(
            pruned.readout[3].weight, source.readout[3].weight[:, expected_indices]))
        removed = [index for index in range(8) if index not in selected]
        with torch.no_grad():
            expected_bias = (source.readout[3].bias
                             + source.readout[3].weight[:, removed]
                             @ activations[:, removed].mean(0))
            logits = pruned.readout(features)
            prediction = pruned(torch.randn(1, 16, 4, 49, 1024))['prediction']
        self.assertTrue(torch.equal(pruned.readout[3].bias, expected_bias))
        self.assertEqual(tuple(logits.shape), (24, 5))
        self.assertTrue(bool(torch.isfinite(logits).all()))
        self.assertEqual(tuple(prediction.shape), (1, 16))
        self.assertTrue(bool(torch.isfinite(prediction).all()))

    def test_historical_checkpoint_without_added_config_keys_roundtrips(self):
        model = self.ready_model()
        value = torch.randn(1, 16, 4, 49, 1024)
        with torch.no_grad():
            expected = model(value)['prediction']
        checkpoint = model.checkpoint()
        checkpoint['config'].pop('fused_readout')
        checkpoint['config'].pop('detector_processing')
        self.assertEqual(checkpoint['schema'], 7)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'historical.pt'
            torch.save(checkpoint, path)
            restored, _ = load_lightgen(path)
        self.assertFalse(restored.cfg.fused_readout)
        self.assertEqual(restored.cfg.detector_processing, 'legacy_log_clip')
        with torch.no_grad():
            actual = restored(value)['prediction']
        self.assertTrue(torch.equal(actual, expected))

    def test_fused_loader_rejects_conflicting_schema_and_configuration(self):
        model = self.ready_model(
            fused_readout=True,
            detector_processing='relative_intensity',
        )
        original = model.checkpoint()
        mutations = (
            ('schema', {'schema': 7}, {}),
            ('processing', {}, {'detector_processing': 'legacy_log_clip'}),
            ('fused', {}, {'fused_readout': False}),
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'conflicting.pt'
            for label, payload_changes, config_changes in mutations:
                with self.subTest(conflict=label):
                    checkpoint = {**original, **payload_changes}
                    checkpoint['config'] = {**original['config'], **config_changes}
                    torch.save(checkpoint, path)
                    with self.assertRaises(ValueError):
                        load_lightgen(path)

    def test_historical_identity_rejects_new_processing_mode(self):
        checkpoint = self.ready_model().checkpoint()
        checkpoint['config']['detector_processing'] = 'relative_intensity'
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'conflicting_historical.pt'
            torch.save(checkpoint, path)
            with self.assertRaises(ValueError):
                load_lightgen(path)


if __name__ == '__main__':
    unittest.main()
