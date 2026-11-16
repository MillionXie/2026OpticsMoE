"""Regression checks for the Google Drive inference-only export."""
from dataclasses import fields
from pathlib import Path
import tempfile
import json
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
            torch.testing.assert_close(actual[name], expected[name], rtol=1e-6, atol=1e-6, msg=name)
        for old, new in (('frame', 'vision'), ('video', 'language')):
            torch.testing.assert_close(actual['routing'][new]['weights'],
                                       expected['routing'][old]['weights'], rtol=1e-6, atol=1e-6)

    def test_frontend_steps_match_the_previous_forward_exactly(self):
        front = public.FrozenVisionFront()
        front.ready.fill_(True)
        front.patch_weight.normal_(std=.01)
        front.patch_bias.normal_(std=.01)
        front.position.normal_(std=.01)
        images = torch.randn(1, 1, 1, 3, 448, 448)
        rgb = images.reshape(-1, 3, 448, 448)
        rgb = rgb.unsqueeze(2).expand(-1, -1, 2, -1, -1).to(front.patch_weight.dtype)
        hidden = torch.nn.functional.conv3d(rgb, front.patch_weight, front.patch_bias,
                                            stride=(2, 16, 16))[:, :, 0]
        hidden = hidden.reshape(-1, 1024, 14, 2, 14, 2)
        hidden = hidden.permute(0, 2, 4, 3, 5, 1).reshape(-1, 784, 1024)
        hidden = hidden + front.position
        merged = hidden.reshape(-1, 196, 4, 1024).mean(2).reshape(-1, 14, 14, 1024)
        expected = merged.reshape(-1, 7, 2, 7, 2, 1024).mean((2, 4))
        expected = expected.reshape(1, 1, 1, 49, 1024).to(torch.float16)
        with torch.inference_mode():
            actual = front(images)
        self.assertTrue(torch.equal(actual, expected))

    def test_propagation_matches_the_formal_temporal_implementation(self):
        from types import SimpleNamespace
        from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.modeling import AngularSpectrum

        config = public.ModelConfig(alpha_eval=.5)
        settings = SimpleNamespace(geometry=SimpleNamespace(canvas_size=config.canvas),
                                   wavelength_nm=config.wavelength_nm, pixel_pitch_um=config.pitch_um,
                                   distance_m=config.distance_m, k_space_enabled=True,
                                   theta_max_deg=config.cutoff_deg)
        reference = AngularSpectrum(settings)
        actual = public.Propagation(config)
        torch.testing.assert_close(actual.transfer, reference.transfer, rtol=1e-7, atol=1e-7)
        field = torch.randn(2, config.canvas, config.canvas, dtype=torch.complex64)
        torch.testing.assert_close(actual(field), reference(field), rtol=1e-6, atol=1e-6)

    def test_propagation_matches_the_notebook_with_the_selected_aperture(self):
        import ast

        path = Path(__file__).resolve().parents[4] / 'opticalmoe/notebooks/github_D2NN_mnist4.ipynb'
        notebook = json.loads(path.read_text('utf-8'))
        cell = next(cell for cell in notebook['cells'] if 'class Propagation_Layer' in ''.join(cell.get('source', [])))
        tree = ast.parse(''.join(cell['source']))
        definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Propagation_Layer')
        namespace = {'torch': torch, 'np': np, 'device': torch.device('cpu')}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), 'exec'), namespace)
        config = public.ModelConfig(alpha_eval=.5)
        reference = namespace['Propagation_Layer'](λ=config.wavelength_nm*1e-9,
                                                 N_pixels=config.canvas,
                                                 pixel_size=config.pitch_um*1e-6,
                                                 distance=config.distance_m)
        actual = public.Propagation(config)
        reference.phase = reference.phase.to(torch.complex64)
        reference.phase = torch.where(actual.phase.abs() > 0, reference.phase, 0)
        self.assertTrue(torch.equal(actual.phase, reference.phase))
        field = torch.randn(3, config.canvas, config.canvas, dtype=torch.complex64)
        self.assertTrue(torch.equal(actual(field), reference(field)))

    def test_propagation_preserves_batch_order_and_input_contract(self):
        propagation = public.Propagation(public.ModelConfig(alpha_eval=.5))
        field = torch.randn(3, 518, 518, dtype=torch.complex64)
        expected = torch.stack([propagation(item) for item in field])
        torch.testing.assert_close(propagation(field), expected, rtol=1e-6, atol=1e-6)
        with self.assertRaisesRegex(ValueError, 'spatial shape'):
            propagation(torch.zeros(2, 517, 518))

    def test_main_saves_metrics_with_python38_pathlib(self):
        from LightGen2 import visualize

        original_write_text = Path.write_text

        def legacy_write_text(path, *args, **kwargs):
            if 'newline' in kwargs:
                raise TypeError("write_text() got an unexpected keyword argument 'newline'")
            return original_write_text(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / 'input.npz'
            np.savez(inputs, format_version=2, sample_ids=np.array(['a', 'b']))
            reference = root / 'reference.csv'
            reference.write_text('sample_id,target_MOS\na,10\nb,20\n', encoding='utf-8')
            arguments = ['infer.py', '--input', str(inputs), '--reference-scores', str(reference),
                         '--device', 'cpu', '--output', str(root)]
            with patch('sys.argv', arguments), \
                    patch.object(public_infer, 'load_lightgen2', return_value=(None, None)), \
                    patch.object(public_infer, 'evaluate', return_value=([], {'videos': 2, 'SRCC': .79})), \
                    patch.object(visualize, 'configure_figures'), \
                    patch.object(Path, 'write_text', new=legacy_write_text):
                public_infer.main()
                public_infer.main()
            metrics = json.loads((root / '02_metrics.json').read_text('utf-8'))
            self.assertEqual(metrics['videos'], 2)
            self.assertEqual(metrics['SRCC'], .79)
            self.assertEqual(set(metrics), {'videos', 'SRCC'})
            self.assertFalse((root / '02_metrics.json.tmp').exists())

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
        archive = {'format_version': np.array(2), 'sample_ids': np.array(['b', 'a'])}
        sample_ids = public_infer.read_input_metadata(archive)
        self.assertEqual(sample_ids, ['b', 'a'])
        archive['format_version'] = np.array(1)
        archive['target_mos'] = np.array([999., -999.])
        self.assertEqual(public_infer.read_input_metadata(archive), sample_ids)
        archive['sample_ids'] = np.array(['a', 'a'])
        with self.assertRaises(ValueError):
            public_infer.read_input_metadata(archive)

    def test_reference_scores_align_by_id_and_reject_invalid_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'reference_mos.csv'
            path.write_text('sample_id,target_MOS\na,10\nb,20\n')
            self.assertEqual(public_infer.read_reference_scores(path, ['b', 'a']).tolist(), [20., 10.])
            for content in ('sample_id,target_MOS\na,10\na,20\n',
                            'sample_id,target_MOS\na,10\nc,20\n',
                            'sample_id,target_MOS\na,nan\nb,20\n'):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    public_infer.read_reference_scores(path, ['b', 'a'])

    def test_reference_labels_affect_only_metrics_and_output_targets(self):
        from LightGen2 import visualize

        class FixedModel:
            def __call__(self, *, normalized_rgb):
                return {'prediction': torch.arange(16, dtype=torch.float32)[None]}

        model = FixedModel()
        inputs = [(0, 2, torch.zeros(1, 16, 4))]
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(public_infer, 'iter_rgb_fields', return_value=inputs), \
                patch.object(visualize, 'capture_optical_fields', side_effect=lambda model, pixels: (model(normalized_rgb=pixels), {})), \
                patch.object(visualize, 'export_evaluation'), \
                patch.object(visualize, 'export_visualizations'):
            first, first_metrics = public_infer.evaluate(model, {}, ['a', 'b'], np.array([10., 20.]), 'cpu', directory)
            second, second_metrics = public_infer.evaluate(model, {}, ['a', 'b'], np.array([20., 10.]), 'cpu', directory)
        self.assertEqual([row['predicted_MOS'] for row in first], [row['predicted_MOS'] for row in second])
        self.assertAlmostEqual(first_metrics['SRCC'], 1.)
        self.assertAlmostEqual(second_metrics['SRCC'], -1.)

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
        self.assertEqual(metrics['videos'], 558)
        self.assertEqual(set(metrics), {'videos', 'SRCC'})
        self.assertTrue(all(set(record) == {'sample_id', 'target_MOS', 'predicted_MOS'}
                            for record in records))
        self.assertEqual([record['sample_id'] for record in records], sample_ids)
        self.assertEqual(export.call_args.args[0], records)
        field_records = fields.call_args.args[1]
        self.assertEqual(len(field_records), 16)
        self.assertEqual(sum(record['is_padding'] for record in field_records), 2)
        self.assertEqual([record['sample_id'] for record in field_records[-2:]], ['0', '1'])


if __name__ == '__main__':
    unittest.main()
