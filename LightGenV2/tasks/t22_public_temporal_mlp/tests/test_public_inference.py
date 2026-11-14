"""Regression checks for the Google Drive inference-only export."""
from dataclasses import fields
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import numpy as np

from LightGen2 import model as public
from LightGen2 import infer as public_infer
from .. import model as research
from ..tools.checkpoint_utils import export_inference_checkpoint, inference_state_name


class PublicInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng_state = torch.get_rng_state()
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(4)
        torch.manual_seed(29)
        config = research.ModelConfig(
            readout_hidden=32, expert_mode='linear', orm_mode='amplitude',
            fusion_router_input='all_tokens', fused_readout=True,
            detector_processing='relative_intensity', shared_global_phase=True)
        cls.original = research.LightGenTemporal(config).eval()
        cls.original.prompt_ready.fill_(True)
        cls.original.quality_ready.fill_(True)
        cls.original.level_scores.copy_(torch.linspace(28.62, 77.30, 5))
        with torch.no_grad():
            cls.original.prompt_embedding.normal_()
        cls.config_fields = [field.name for field in fields(public.ModelConfig)]
        public_config = public.ModelConfig(**{name: getattr(config, name) for name in cls.config_fields})
        cls.inference = public.LightGen2(public_config).eval()
        cls.omitted = {'target_mean', 'target_std', 'level_boundaries'}
        state = {inference_state_name(name): value for name, value in cls.original.state_dict().items()
                 if name not in cls.omitted}
        cls.inference.load_state_dict(state, strict=True)
        cls.input = torch.randn(1, 16, 4, 49, 1024)

    @classmethod
    def tearDownClass(cls):
        torch.set_rng_state(cls.rng_state)
        torch.set_num_threads(cls.threads)

    def test_forward_matches_research_inference(self):
        with torch.inference_mode():
            expected = self.original(self.input)
            actual = self.inference.predict_from_embeddings(self.input)
        for name in ('prediction', 'quality_logits', 'quality_probabilities'):
            self.assertTrue(torch.equal(expected[name], actual[name]), name)
        for old, new in (('frame', 'vision'), ('video', 'language')):
            self.assertTrue(torch.equal(expected['routing'][old]['weights'],
                                        actual['routing'][new]['weights']))

    def test_export_runs_in_inference_mode_and_retains_tensors(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'source.pt'
            destination = Path(directory)/'LightGen2.pt'
            torch.save(self.original.checkpoint(), source)
            identity = export_inference_checkpoint(source, destination, self.config_fields, public.ARCHITECTURE)
            loaded, _ = public.load_lightgen2(destination)
            self.assertFalse(loaded.predict_from_embeddings(self.input)['prediction'].requires_grad)
            self.assertEqual(set(identity['omitted_auxiliary_buffers']), self.omitted)
            self.assertTrue(identity['retained_model_tensors_bitwise_equal'])
            expected_names = {inference_state_name(name) for name in self.original.state_dict()
                              if name not in self.omitted}
            self.assertEqual(set(loaded.state_dict()), expected_names)
            self.assertTrue(torch.equal(loaded.predict_from_embeddings(self.input)['prediction'],
                                        self.inference.predict_from_embeddings(self.input)['prediction']))

    def test_profile_dimensions_are_bound_to_selected_architecture(self):
        with self.assertRaises(ValueError):
            public.ModelConfig(alpha_eval=.5, readout_hidden=128)
        with self.assertRaises(ValueError):
            public.ModelConfig(alpha_eval=float('nan'))

    def test_published_model_has_forward_components(self):
        self.assertFalse(hasattr(self.inference, 'set_quality_levels'))
        self.assertFalse(hasattr(self.inference, 'checkpoint'))
        self.assertFalse(self.omitted.intersection(self.inference.state_dict()))
        self.assertEqual(sum(parameter.numel() for parameter in self.inference.parameters()), 3407720)

    def test_normalized_rgb_groups_preserve_order_and_tail_padding(self):
        first = np.empty((16, 4, 3, 448, 448), dtype=np.float16)
        for index in range(16):
            first[index].fill(index / 16)
        last = np.full((2, 4, 3, 448, 448), .5, dtype=np.float16)
        archive = {'rgb_0001': first, 'rgb_0002': last}
        fields = list(public_infer.iter_rgb_fields(archive, [str(i) for i in range(18)]))
        self.assertEqual([(start, count) for start, count, _ in fields], [(0, 16), (16, 2)])
        self.assertEqual(fields[-1][2].shape, (1, 16, 4, 3, 448, 448))
        self.assertEqual(fields[-1][2][0, 2:, 0, 0, 0, 0].tolist(),
                         [index / 16 for index in range(14)])

    def test_npz_metadata_preserves_order_and_rejects_duplicate_ids(self):
        archive = {'format_version': np.array(1), 'sample_ids': np.array(['b', 'a']),
                   'target_mos': np.array([20., 10.])}
        sample_ids, targets = public_infer.read_input_metadata(archive)
        self.assertEqual(sample_ids, ['b', 'a'])
        self.assertEqual(targets.tolist(), [20., 10.])
        archive['sample_ids'] = np.array(['a', 'a'])
        with self.assertRaises(ValueError):
            public_infer.read_input_metadata(archive)

    def test_normalized_rgb_storage_preserves_patch_projection_precision(self):
        pixels = torch.arange(256).float()
        original = ((pixels / 255 - .5) / .5).to(torch.bfloat16)
        stored = original.to(torch.float16)
        self.assertTrue(torch.equal(stored.to(torch.bfloat16), original))
        self.assertTrue(torch.equal(((stored.float() + 1) * 127.5).round(), pixels))

    def test_full_evaluation_scores_each_field_once_and_excludes_padding(self):
        from LightGen2 import visualize

        class CountingModel:
            calls = 0

            def __call__(self, *, normalized_rgb):
                self.calls += 1
                selected = torch.tensor([True, True, False, False]).expand(1, 16, 4)
                probabilities = torch.full((1, 16, 4), .25)
                return {'prediction': normalized_rgb[:, :, 0].float() + 1000 * self.calls,
                        'routing': {'vision': {'selected': selected[:, :, None].expand(1, 16, 4, 4),
                                               'probabilities': probabilities[:, :, None].expand(1, 16, 4, 4)},
                                    'language': {'selected': selected, 'probabilities': probabilities}}}

        model = CountingModel()
        sample_ids = [str(index) for index in range(558)]
        targets = np.arange(558)

        def fields(archive, ids):
            for start in range(0, len(ids), 16):
                count = min(16, len(ids) - start)
                indices = list(range(start, start + count)) + list(range(16 - count))
                yield start, count, torch.tensor(indices)[None, :, None]

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(public_infer, 'iter_rgb_fields', side_effect=fields), \
                patch.object(visualize, 'capture_optical_fields', side_effect=lambda model, pixels: (model(normalized_rgb=pixels), {})) as capture, \
                patch.object(visualize, 'export_evaluation') as export, \
                patch.object(visualize, 'export_visualizations') as fields:
            records, metrics = public_infer.evaluate(model, {}, sample_ids, targets, 'cpu', directory,
                                                    visualize_field=35)
        self.assertEqual(model.calls, 35)
        self.assertEqual(capture.call_count, 1)
        self.assertEqual(len(records), 558)
        self.assertEqual(metrics['fields'], 35)
        self.assertEqual(metrics['videos'], 558)
        self.assertEqual([record['sample_id'] for record in records], sample_ids)
        self.assertEqual(export.call_args.args[2]['vision']['selected'].shape[0], 558)
        field_records = fields.call_args.args[1]
        self.assertEqual(len(field_records), 16)
        self.assertEqual(sum(record['is_padding'] for record in field_records), 2)
        self.assertEqual([record['sample_id'] for record in field_records[-2:]], ['0', '1'])


if __name__ == '__main__':
    unittest.main()
