"""Keep each branch's learned mask identical across all video slots."""
from dataclasses import fields
from pathlib import Path
import tempfile
import unittest

import torch

from LightGen2 import model as public
from ..model import LightGenTemporal, ModelConfig, load_lightgen
from ..tools.analyze_phase_effect import constant_modulation, phase_information
from ..tools.checkpoint_utils import export_inference_checkpoint, share_global_phases


class SharedGlobalPhaseTests(unittest.TestCase):
    def setUp(self):
        self.random_state = torch.get_rng_state()
        self.threads = torch.get_num_threads()
        torch.set_num_threads(4)
        torch.manual_seed(814)
        config = ModelConfig(readout_hidden=32, expert_mode='linear', orm_mode='amplitude',
                             fusion_router_input='all_tokens', fused_readout=True,
                             detector_processing='relative_intensity')
        self.parent = LightGenTemporal(config).eval()

    def tearDown(self):
        torch.set_rng_state(self.random_state)
        torch.set_num_threads(self.threads)

    def test_conversion_preserves_all_other_tensors_and_uses_one_parameter(self):
        shared = share_global_phases(self.parent)
        self.assertTrue(shared.cfg.shared_global_phase)
        phase_names = {'frame_global.raw_global_phase', 'video_global.raw_global_phase'}
        for name, value in self.parent.state_dict().items():
            if name in phase_names:
                self.assertEqual(shared.state_dict()[name].shape, (1, 111, 111))
            else:
                self.assertTrue(torch.equal(value, shared.state_dict()[name]), name)
        self.assertIs(share_global_phases(shared), shared)

    def test_optimizer_updates_one_shared_mask_for_all_slots(self):
        shared = share_global_phases(self.parent)
        for stage in (shared.frame_global, shared.video_global):
            optimizer = torch.optim.SGD([stage.raw_global_phase], lr=.02)
            before = stage.raw_global_phase.detach().clone()
            weights = torch.arange(1, 17).reshape(16, 1, 1)
            (stage.phase() * weights).mean().backward()
            self.assertEqual(stage.raw_global_phase.grad.shape, (1, 111, 111))
            optimizer.step()
            self.assertFalse(torch.equal(before, stage.raw_global_phase))
            phase = stage.phase()
            self.assertEqual(phase.stride(0), 0)
            for slot in range(16):
                self.assertTrue(torch.equal(phase[0], phase[slot]))
                self.assertTrue(torch.equal(stage.modulation()[0], stage.modulation()[slot]))

    def test_shared_and_legacy_checkpoint_round_trips(self):
        shared = share_global_phases(self.parent)
        for model, schema, tile_count in ((shared, 9, 1), (self.parent, 8, 16)):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'model.pt'
                checkpoint = model.checkpoint(epoch=0)
                self.assertEqual(checkpoint['schema'], schema)
                torch.save(checkpoint, path)
                restored, metadata = load_lightgen(path)
                self.assertEqual(metadata['epoch'], 0)
                self.assertEqual(restored.frame_global.raw_global_phase.shape[0], tile_count)
                for name, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value, restored.state_dict()[name]), name)

    def test_constant_phase_control_keeps_all_sixteen_slots(self):
        shared = share_global_phases(self.parent)
        for stage in (shared.frame_global, shared.video_global):
            _, _, centers, _ = phase_information(stage)
            constant = constant_modulation(stage, centers)
            self.assertEqual(constant.shape, (16, 111, 111))
            self.assertTrue(torch.equal(constant, constant[:1].expand_as(constant)))

    def test_public_loader_retains_legacy_checkpoint_support(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.pt'
            destination = Path(directory) / 'legacy.pt'
            torch.save(self.parent.checkpoint(), source)
            config_fields = [field.name for field in fields(public.ModelConfig)]
            export_inference_checkpoint(source, destination, config_fields, public.LEGACY_ARCHITECTURE)
            restored, _ = public.load_lightgen2(destination)
            self.assertFalse(restored.cfg.shared_global_phase)
            self.assertEqual(restored.vision_global.raw_phase.shape, (16, 111, 111))


if __name__ == '__main__':
    unittest.main()
