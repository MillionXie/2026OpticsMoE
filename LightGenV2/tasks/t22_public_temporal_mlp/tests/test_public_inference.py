"""Regression checks for the Google Drive inference-only export."""
from dataclasses import fields
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

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
            detector_processing='relative_intensity')
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
        self.assertEqual(sum(parameter.numel() for parameter in self.inference.parameters()), 3777350)

    def test_558_video_grouping_retains_order_and_two_tail_padding_entries(self):
        paths = [Path(str(index)) for index in range(558)]
        with patch.object(public_infer, 'frames_from_video', side_effect=lambda path: torch.tensor([int(path.name)])):
            fields = list(public_infer.iter_video_fields(paths))
        self.assertEqual(len(fields), 35)
        self.assertEqual(len(fields[-1][0]), 14)
        self.assertEqual(fields[-1][1][0, :, 0].tolist(), list(range(544, 558)) + [0, 1])
        self.assertEqual([path for chunk, _ in fields for path in chunk], paths)

    def test_evaluation_csv_preserves_order_and_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('a.mp4', 'b.mp4'):
                (root / name).touch()
            path = root / 'test.csv'
            path.write_text('sample_id,video_path,mos\nb,b.mp4,20\na,a.mp4,10\n', encoding='utf-8')
            rows, videos, targets = public_infer.read_evaluation_csv(path, root)
            self.assertEqual([row['sample_id'] for row in rows], ['b', 'a'])
            self.assertEqual([video.name for video in videos], ['b.mp4', 'a.mp4'])
            self.assertEqual(targets.tolist(), [20, 10])
            path.write_text('sample_id,video_path,mos\na,b.mp4,20\na,a.mp4,10\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                public_infer.read_evaluation_csv(path, root)

    def test_demo_subset_reads_only_requested_video_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'a.mp4').touch()
            (root / 'b.mp4').touch()
            path = root / 'test.csv'
            path.write_text('sample_id,video_path,mos\na,a.mp4,10\nb,b.mp4,20\nc,absent.mp4,30\n', encoding='utf-8')
            rows, videos, targets = public_infer.read_evaluation_csv(path, root, limit=2)
            self.assertEqual([row['sample_id'] for row in rows], ['a', 'b'])
            self.assertEqual([video.name for video in videos], ['a.mp4', 'b.mp4'])
            self.assertEqual(targets.tolist(), [10, 20])


if __name__ == '__main__':
    unittest.main()
