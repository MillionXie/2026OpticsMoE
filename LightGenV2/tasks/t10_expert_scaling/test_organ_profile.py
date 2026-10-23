"""CPU contracts for orientation-safe augmentation and noise-free evaluation."""
import copy
import json
from pathlib import Path
import unittest

import torch

from .model import ScalingOptics
from .plan import load_protocol
from .train import orientation_augmentation


class OrganProfileTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.profile=json.loads((Path(__file__).parent/'configs/organ_orientation_regularized_s17.json').read_text())

    def test_augmentation_is_deterministic_chunk_independent_and_grayscale(self):
        gray=torch.linspace(0,1,2*28*28).reshape(2,1,28,28).repeat(1,3,1,1)
        cfg=self.profile['augmentation'];idx=torch.tensor([7,12])
        full=orientation_augmentation(gray,idx,3,17,cfg)
        chunks=torch.cat([orientation_augmentation(gray[i:i+1],idx[i:i+1],3,17,cfg) for i in range(2)])
        self.assertTrue(torch.allclose(full,chunks))
        self.assertTrue(torch.equal(full[:,0],full[:,1]))
        self.assertTrue(torch.isfinite(full).all() and full.min()>=0 and full.max()<=1)
        self.assertFalse(torch.equal(full,orientation_augmentation(gray,idx,4,17,cfg)))

    def test_zero_strength_is_identity_and_no_reflection(self):
        cfg=dict(self.profile['augmentation'],rotation_degrees=0,translation_fraction=0,
                 scale_min=1,scale_max=1,contrast_delta=0,brightness_delta=0,gamma_min=1,gamma_max=1)
        x=torch.rand(2,3,28,28)
        self.assertTrue(torch.allclose(x,orientation_augmentation(x,torch.tensor([1,2]),1,17,cfg),atol=3e-6))

    def test_phase_noise_disabled_in_eval_without_changing_checkpoint_shape(self):
        cfg=load_protocol();clean=ScalingOptics(cfg,4,4,'moe_oeo',11,4,2,checkpointing=False)
        noisy_cfg=copy.deepcopy(cfg);noisy_cfg['training_proposal']['phase_noise_std']=.02
        noisy=ScalingOptics(noisy_cfg,4,4,'moe_oeo',11,4,2,checkpointing=False)
        noisy.load_state_dict(clean.state_dict());clean.eval();noisy.eval()
        x=torch.rand(1,3,28,28)
        with torch.no_grad():
            self.assertTrue(torch.equal(clean(x)['probabilities'],noisy(x)['probabilities']))


if __name__ == '__main__':unittest.main()
