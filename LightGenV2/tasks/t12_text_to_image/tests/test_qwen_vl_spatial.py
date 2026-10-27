import unittest
from types import SimpleNamespace

import torch
from torch import nn
from diffusers import EulerDiscreteScheduler

from LightGenV2.tasks.t12_text_to_image.qwen_vl_spatial import (
    SpatialBridge,SpatialInputConv,SpatialTurboHead,balanced_region_l1)
from LightGenV2.tasks.t12_text_to_image.qwen_vl_turbo_model import NativeTurboHead,QueryResampler


class UNet(nn.Module):
    dtype=torch.float32
    def __init__(self):super().__init__();self.conv_in=nn.Conv2d(4,4,3,padding=1)
    def forward(self,x,t,encoder_hidden_states):
        return SimpleNamespace(sample=self.conv_in(x)+encoder_hidden_states.mean((1,2))[:,None,None,None]*.001)


class VAE(nn.Module):
    dtype=torch.float32
    config=SimpleNamespace(scaling_factor=1.)
    def encode(self,*a):raise AssertionError('Inference reference VAE encoder must not be used')
    def decode(self,x):return SimpleNamespace(sample=x[:,:3])


class SpatialTests(unittest.TestCase):
    def test_zero_injection_preserves_entire_native_step(self):
        torch.manual_seed(5)
        unet=UNet();vae=VAE();adapter=QueryResampler(16,16,8,4,1)
        scheduler=EulerDiscreteScheduler(timestep_spacing='trailing')
        h=torch.randn(2,7,16);valid=torch.ones(2,7,dtype=torch.bool);noise=torch.randn(2,4,8,8)
        base=NativeTurboHead(unet,vae,scheduler,adapter).eval()
        expected=base(h,valid,noise)
        unet.conv_in=SpatialInputConv(unet.conv_in,16)
        model=SpatialTurboHead(unet,vae,scheduler,adapter,SpatialBridge(16,16,16,2),[1,2,3,4]).eval()
        actual=model(h,valid,noise)
        for a,b in zip(actual,expected):torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_spatial_path_learns_without_modifying_noise_convolution(self):
        torch.manual_seed(6)
        original=nn.Conv2d(4,4,3,padding=1);initial=original.weight.detach().clone()
        conv=SpatialInputConv(original,16);bridge=SpatialBridge(16,16,16,2)
        optimizer=torch.optim.SGD(list(bridge.parameters())+list(conv.spatial.parameters()),lr=.01)
        hidden=torch.randn(2,4,16);noise=torch.randn(2,4,8,8)
        for _ in range(2):
            spatial=bridge(hidden);loss=conv(torch.cat([noise,spatial],1)).square().mean()
            optimizer.zero_grad();loss.backward();optimizer.step()
        self.assertIsNone(original.weight.grad)
        torch.testing.assert_close(original.weight,initial,atol=0,rtol=0)
        self.assertGreater(bridge.project.weight.grad.abs().sum().item(),0)
        self.assertGreater(conv.spatial.weight.abs().sum().item(),0)

    def test_bad_grid_and_padding_are_rejected(self):
        bridge=SpatialBridge(16,16,16,2)
        with self.assertRaises(ValueError):bridge(torch.randn(1,3,16))
        model=SpatialTurboHead(UNet(),VAE(),None,None,bridge,[1,2,3,4])
        with self.assertRaises(ValueError):model.visual_tokens(torch.randn(1,6,16),torch.tensor([[1,1,1,1,0,0]],dtype=torch.bool))

    def test_region_balance_does_not_hide_small_edit(self):
        pred=torch.zeros(1,3,4,4,requires_grad=True);target=torch.zeros_like(pred)
        target[:,:,0,0]=1;region=torch.zeros(1,1,4,4);region[:,:,0,0]=1
        loss=balanced_region_l1(pred,target,region)
        torch.testing.assert_close(loss,torch.tensor(.5));loss.backward()
        self.assertGreater(pred.grad.abs().sum().item(),0)
        torch.testing.assert_close(balanced_region_l1(pred.detach(),target,torch.ones_like(region)),
                                   (pred.detach()-target).abs().mean())


if __name__=='__main__':unittest.main()
