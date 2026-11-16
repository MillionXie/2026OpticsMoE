from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from ..model import LightGenTemporal, ModelConfig, load_lightgen
from ..tools.router_power_candidate import (FixedInputScaleGlobal, encoded_amplitude,
                                             make_candidate, load_candidate)


class RouterPowerCandidateTests(unittest.TestCase):
    def config(self):
        return ModelConfig(readout_hidden=32,expert_mode='linear',orm_mode='amplitude',
                           fusion_router_input='all_tokens',fused_readout=True,
                           detector_processing='relative_intensity',shared_global_phase=True)

    def test_only_fixed_scale_buffers_are_added_to_the_parent_state(self):
        parent = LightGenTemporal(self.config())
        candidate = make_candidate(parent)
        extras = set(candidate.state_dict())-set(parent.state_dict())
        self.assertEqual(extras,{'frame_global.input_scale','video_global.input_scale'})
        for name,value in parent.state_dict().items():
            self.assertTrue(torch.equal(value,candidate.state_dict()[name]),name)

    def test_fixed_input_scale_preserves_region_energy_differences(self):
        stage = FixedInputScaleGlobal(self.config(),False)
        with torch.no_grad():
            stage.width_to_field.weight.fill_(1/192)
            stage.width_to_field.bias.zero_()
            stage.input_scale.fill_(2)
        values = torch.ones(1,2,42,192)
        values[:,1].mul_(3)
        raw = encoded_amplitude(stage,values)
        fields = stage.fields(values)
        self.assertTrue(torch.equal(fields,raw/2))
        energies = fields.square().sum((-2,-1))
        torch.testing.assert_close(energies[0,1],9*energies[0,0],rtol=3e-6,atol=1e-5)

    def test_candidate_has_a_distinct_checkpoint_identity_and_reloads_its_scales(self):
        model = make_candidate(LightGenTemporal(self.config()))
        model.frame_global.input_scale.fill_(.3)
        model.video_global.input_scale.fill_(.7)
        with TemporaryDirectory() as directory:
            path = Path(directory)/'candidate.pt'
            torch.save(model.checkpoint(role='candidate'),path)
            loaded,_ = load_candidate(path)
            for name,value in model.state_dict().items():
                self.assertTrue(torch.equal(value,loaded.state_dict()[name]),name)
            with self.assertRaises(ValueError):
                load_lightgen(path)


if __name__ == '__main__':
    unittest.main()
