import unittest
from unittest.mock import patch
import torch
from ..scene_resnet_supervised_train import fresh_readout,image_epsilon_loss,tensor_state_sha,PROFILE
from ..scene_resnet_frontend import ResNetConditionReadout,experiment_profile,ELECTRONIC


class IndependentResNetTests(unittest.TestCase):
    def test_fresh_readout_never_uses_inherited_parameters(self):
        with patch.object(ResNetConditionReadout,'initialize_readouts',side_effect=AssertionError('Forbidden inheritance')):
            first=fresh_readout();second=fresh_readout()
        self.assertEqual(tensor_state_sha(first),tensor_state_sha(second))
        self.assertEqual(sum(p.numel()for p in first.parameters()),7062496)
        profile=experiment_profile(PROFILE)
        self.assertIsNone(profile['teacher']);self.assertEqual(profile['steps'],30)
        self.assertEqual(profile['generator_sha256'],ELECTRONIC)

    def test_reviewed_weights_and_accumulated_batch_match(self):
        torch.manual_seed(42)
        pred=torch.randn(8,4,3,3,requires_grad=True);target=torch.randn_like(pred,requires_grad=True)
        weights=torch.tensor([.5,1.,1.,.5,.5,1.,.5,1.])
        full=image_epsilon_loss(pred,target,weights);expected=torch.autograd.grad(full,pred,retain_graph=True)[0]
        pieces=sum(image_epsilon_loss(pred[i:i+2],target[i:i+2],weights[i:i+2])*2/8 for i in range(0,8,2))
        actual=torch.autograd.grad(pieces,pred)[0]
        self.assertTrue(torch.allclose(full,pieces));self.assertTrue(torch.equal(expected,actual))
        full.backward();self.assertIsNone(target.grad)
        with self.assertRaises(ValueError):image_epsilon_loss(pred,target,torch.zeros(8))

    def test_frozen_decoder_allows_frontend_image_gradient(self):
        frontend=torch.nn.Linear(2,4);decoder=torch.nn.Linear(4,4).requires_grad_(False)
        before=tensor_state_sha(decoder)
        value=decoder(frontend(torch.ones(2,2))).reshape(2,1,2,2)
        image_epsilon_loss(value,torch.zeros_like(value),torch.tensor([.5,1.])).backward()
        self.assertGreater(float(frontend.weight.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in decoder.parameters()))
        self.assertEqual(before,tensor_state_sha(decoder))


if __name__=='__main__':unittest.main()
