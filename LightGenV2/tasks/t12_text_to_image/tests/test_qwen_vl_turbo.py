"""Synthetic protocol and native scheduler/gradient checks; no real TEST data."""
import copy
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from diffusers import EulerDiscreteScheduler

from LightGenV2.tasks.t12_text_to_image.qwen_vl_turbo_data import validate_roles, pilot_subset
from LightGenV2.tasks.t12_text_to_image.qwen_vl_turbo_model import NativeTurboHead, QueryResampler, pack_hidden, region_weighted_loss


def row(split, mode, source, target):
    def product(identity):
        return {'product_id':identity,'family_id':identity,'rgba_sha256':identity+'-pixels'}
    return {'split':split,'mode':mode,'category':'lamp','source':product(source),'target':product(target)}


class ProtocolTests(unittest.TestCase):
    def test_target_role_leakage_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'Cross-split'):
            validate_roles([row('train','object','a','shared'),row('val','object','v','shared')])

    def test_family_and_pixels_are_also_grouped(self):
        for field in ('family_id','rgba_sha256'):
            a,b=row('train','background','a','a'),row('val','background','b','b')
            b['target'][field]=a['target'][field]
            with self.assertRaises(ValueError): validate_roles([a,b])

    def test_test_and_identity_edits_rejected(self):
        for value in (row('test','background','a','a'),row('train','object','a','a')):
            with self.assertRaises(ValueError): validate_roles([value])

    def test_balanced_pilot_selects_distinct_products(self):
        rows=[row('train',mode,source,'t') for mode in ('background','object','joint')
              for source in ('a','a','b','c')]
        self.assertEqual(len(pilot_subset(rows,2)),6)


class DummyUNet(nn.Module):
    dtype=torch.float32
    def forward(self,x,t,encoder_hidden_states):
        return SimpleNamespace(sample=x*.01+encoder_hidden_states.mean((1,2))[:,None,None,None])


class DummyVAE(nn.Module):
    dtype=torch.float32
    config=SimpleNamespace(scaling_factor=.18215)
    def encode(self,*args):
        raise AssertionError('Reference VAE encoding is forbidden in forward')
    def decode(self,x):
        return SimpleNamespace(sample=x[:,:3])


class NativeHeadTests(unittest.TestCase):
    def test_foreground_supervision_is_normalized_and_differentiable(self):
        a=torch.zeros(1,3,4,4,requires_grad=True); b=torch.zeros_like(a)
        b[:,:,0,0]=1
        region=torch.zeros(1,1,4,4); region[:,:,0,0]=1
        uniform=region_weighted_loss(a,b,region,0)
        torch.testing.assert_close(uniform,(a-b).square().mean())
        focused=region_weighted_loss(a,b,region,5)
        self.assertGreater(focused.item(),uniform.item())
        focused.backward()
        self.assertTrue(torch.isfinite(a.grad).all())

    def test_padding_does_not_affect_condition(self):
        torch.manual_seed(1)
        adapter=QueryResampler(input_dim=16,hidden=16,output_dim=8,queries=4,layers=1).eval()
        a=torch.randn(1,3,16)
        padded=torch.cat([a,torch.randn(1,2,16)*100],1)
        torch.testing.assert_close(adapter(a,torch.ones(1,3,dtype=torch.bool)),
                                   adapter(padded,torch.tensor([[1,1,1,0,0]],dtype=torch.bool)),atol=1e-6,rtol=1e-5)

    def test_native_step_is_repeatable_and_differentiable(self):
        torch.manual_seed(2)
        adapter=QueryResampler(input_dim=16,hidden=16,output_dim=8,queries=4,layers=1)
        scheduler=EulerDiscreteScheduler(prediction_type='epsilon',timestep_spacing='trailing')
        model=NativeTurboHead(DummyUNet(),DummyVAE(),scheduler,adapter)
        h=torch.randn(1,3,16); valid=torch.ones(1,3,dtype=torch.bool); noise=torch.randn(1,4,4,4)
        z,rgb=model(h,valid,noise)
        z2,_=model(h,valid,noise)
        torch.testing.assert_close(z,z2)
        independent=EulerDiscreteScheduler.from_config(scheduler.config)
        independent.set_timesteps(1)
        t=independent.timesteps[0]; latent=noise*independent.init_noise_sigma
        prediction=model.unet(independent.scale_model_input(latent,t),t,adapter(h,valid)).sample
        expected=independent.step(prediction,t,latent).prev_sample
        torch.testing.assert_close(z,expected)
        rgb.square().mean().backward()
        self.assertGreater(adapter.project.weight.grad.abs().sum().item(),0)
        torch.testing.assert_close(pack_hidden([h[0]],'cpu')[0],h)


if __name__=='__main__': unittest.main()
