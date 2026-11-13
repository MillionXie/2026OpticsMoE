"""Regression checks for the Google Drive inference-only export."""
from dataclasses import fields
from pathlib import Path
import tempfile
import unittest

import torch

from LightGen2 import model as public
from .. import model as research
from ..tools.checkpoint_utils import export_inference_checkpoint


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
        cls.inference = public.LightGenTemporal(public_config).eval()
        cls.omitted = {'target_mean', 'target_std', 'level_boundaries'}
        state = {name: value for name, value in cls.original.state_dict().items() if name not in cls.omitted}
        cls.inference.load_state_dict(state, strict=True)
        cls.input = torch.randn(1, 16, 4, 49, 1024)

    @classmethod
    def tearDownClass(cls):
        torch.set_rng_state(cls.rng_state)
        torch.set_num_threads(cls.threads)

    def test_forward_matches_research_inference(self):
        with torch.inference_mode():
            expected = self.original(self.input)
            actual = self.inference(self.input)
        for name in ('prediction', 'quality_logits', 'quality_probabilities'):
            self.assertTrue(torch.equal(expected[name], actual[name]), name)
        for stage in ('frame', 'video'):
            self.assertTrue(torch.equal(expected['routing'][stage]['weights'],
                                        actual['routing'][stage]['weights']))

    def test_export_runs_in_inference_mode_and_retains_tensors(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'source.pt'
            destination = Path(directory)/'LightGen.pt'
            torch.save(self.original.checkpoint(), source)
            identity = export_inference_checkpoint(source, destination, self.config_fields, public.ARCHITECTURE)
            loaded, _ = public.load_lightgen(destination)
            self.assertFalse(loaded(self.input)['prediction'].requires_grad)
            self.assertEqual(set(identity['omitted_auxiliary_buffers']), self.omitted)
            self.assertTrue(identity['retained_model_tensors_bitwise_equal'])
            self.assertEqual(set(loaded.state_dict()), set(self.original.state_dict())-self.omitted)
            self.assertTrue(torch.equal(loaded(self.input)['prediction'], self.inference(self.input)['prediction']))

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


if __name__ == '__main__':
    unittest.main()
