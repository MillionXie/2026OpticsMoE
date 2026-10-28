import unittest
from types import SimpleNamespace

import torch
from torch import nn
from diffusers import EulerDiscreteScheduler

from LightGenV2.tasks.t12_text_to_image.qwen_vl_spatial import (
    SpatialBridge,SpatialInputConv,SpatialTurboHead,balanced_region_l1)
from LightGenV2.tasks.t12_text_to_image.qwen_vl_turbo_model import NativeTurboHead,QueryResampler
from LightGenV2.tasks.t12_text_to_image.qwen_vl_spatial_run import category_ids,lr_factor,selection_score


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
    def test_counterfactuals_only_use_different_train_instructions_on_identical_rgb(self):
        from ..qwen_vl_spatial_run import counterfactual_negatives
        rows=[{'split':'train','render_protocol':'rich_product_scene_v1','source':{'sample_id':'one'},'mode':'object','prompt':p} for p in ('red','blue')]
        data={'rows':rows,'reference':torch.zeros(2,3,4,4,dtype=torch.uint8)}
        self.assertEqual(counterfactual_negatives(data,[0,1]),{0:[1],1:[0]})
        self.assertEqual(counterfactual_negatives(data,[0]),{0:[1]})
        rows[1]['split']='val'
        with self.assertRaises(ValueError):counterfactual_negatives(data,[0,1])
        rows[1]['split']='train';data['reference'][1]+=1
        with self.assertRaises(ValueError):counterfactual_negatives(data,[0,1])

    def test_conditional_spatial_identity_then_text_control_without_reference_aux_corruption(self):
        torch.manual_seed(3)
        old=SpatialBridge(16,16,4,2)
        conditioned=SpatialBridge(16,16,4,2,conditional=True)
        missing,extra=conditioned.load_state_dict(old.state_dict(),strict=False)
        self.assertEqual(set(missing),{'film.weight','film.bias'});self.assertFalse(extra)
        visual=torch.randn(2,4,16);text=torch.randn(2,16,requires_grad=True)
        original,aux=old(visual,with_aux=True)
        actual,actual_aux=conditioned(visual,with_aux=True,context=text)
        torch.testing.assert_close(actual,original,atol=0,rtol=0)
        torch.testing.assert_close(actual_aux,aux,atol=0,rtol=0)
        with torch.no_grad():conditioned.film.weight.normal_(std=.03)
        edited,unmodified_aux=conditioned(visual,with_aux=True,context=text)
        self.assertFalse(torch.equal(edited,original))
        torch.testing.assert_close(unmodified_aux,aux,atol=0,rtol=0)
        edited.square().mean().backward();self.assertGreater(float(text.grad.abs().sum()),0.)
        with self.assertRaises(ValueError):conditioned(visual)

    def test_two_category_scope_preserves_all_modes_and_excludes_other_rows(self):
        rows=[{'category':cat,'split':split,'mode':mode} for cat in ('lamp','pillow','ottoman')
              for split in ('train','val') for mode in ('background','object','joint')]
        ids=category_ids(rows,['lamp','pillow'])
        self.assertEqual(len(ids),12)
        self.assertTrue(all(rows[i]['category']!='ottoman' for i in ids))
        with self.assertRaises(ValueError):category_ids(rows,['chair'])
        with self.assertRaises(ValueError):category_ids(rows[:-1],['ottoman'])

    def test_perceptual_selection_and_decay_direction(self):
        smooth={'mean_lpips':.4,'mean_per_image_psnr_db':30}
        detailed={'mean_lpips':.2,'mean_per_image_psnr_db':28}
        self.assertGreater(selection_score(detailed,'mean_lpips'),selection_score(smooth,'mean_lpips'))
        self.assertGreater(selection_score(smooth,'mean_per_image_psnr_db'),selection_score(detailed,'mean_per_image_psnr_db'))
        self.assertGreater(selection_score({'mean_lpips':.3,'mean_object_crop_lpips':.2},'mean_lpips_plus_object_crop'),
                           selection_score({'mean_lpips':.3,'mean_object_crop_lpips':.4},'mean_lpips_plus_object_crop'))
        self.assertEqual(lr_factor(36,1800,36,.1),1)
        self.assertAlmostEqual(lr_factor(1800,1800,36,.1),.1)
        self.assertLess(lr_factor(1,1800,36,.1),lr_factor(20,1800,36,.1))
        self.assertGreater(lr_factor(100,1800,36,.1),lr_factor(1000,1800,36,.1))

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

    def test_native_prefix_does_not_change_semantic_conditioning(self):
        torch.manual_seed(8)
        unet=UNet();vae=VAE();adapter=QueryResampler(16,16,8,4,1)
        scheduler=EulerDiscreteScheduler(timestep_spacing='trailing')
        fused=torch.randn(2,7,16);valid=torch.ones(2,7,dtype=torch.bool);noise=torch.randn(2,4,8,8)
        base=NativeTurboHead(unet,vae,scheduler,adapter).eval();expected=base(fused,valid,noise)
        unet.conv_in=SpatialInputConv(unet.conv_in,16)
        model=SpatialTurboHead(unet,vae,scheduler,adapter,SpatialBridge(16,16,16,2),[0,1,2,3],4).eval()
        h=torch.cat([torch.randn(2,4,16),fused],1);v=torch.ones(2,11,dtype=torch.bool)
        for a,b in zip(model(h,v,noise),expected):torch.testing.assert_close(a,b,atol=0,rtol=0)
        torch.testing.assert_close(model.semantic_conditions(h,v),adapter(fused,valid),atol=0,rtol=0)

    def test_bad_grid_and_padding_are_rejected(self):
        bridge=SpatialBridge(16,16,16,2)
        with self.assertRaises(ValueError):bridge(torch.randn(1,3,16))
        model=SpatialTurboHead(UNet(),VAE(),None,None,bridge,[1,2,3,4])
        with self.assertRaises(ValueError):model.visual_tokens(torch.randn(1,6,16),torch.tensor([[1,1,1,1,0,0]],dtype=torch.bool))

    def test_focused_readout_only_uses_verified_instruction_span(self):
        torch.manual_seed(5)
        adapter=QueryResampler(16,16,8,4,1)
        model=SpatialTurboHead(UNet(),VAE(),None,adapter,SpatialBridge(16,16,16,2),[1,2,3,4]).eval()
        model.instruction_span=(6,2);model.text_readout_gain=.15
        h=torch.randn(2,12,16);valid=torch.ones(2,12,dtype=torch.bool)
        mask=valid.clone();mask[:,:6]=False;mask[:,10:]=False
        torch.testing.assert_close(model.semantic_conditions(h,valid),adapter(h,mask),atol=0,rtol=0)
        altered=h.clone();altered[:,:6]+=100
        torch.testing.assert_close(model.semantic_conditions(altered,valid),model.semantic_conditions(h,valid),atol=0,rtol=0)

    def test_region_balance_does_not_hide_small_edit(self):
        pred=torch.zeros(1,3,4,4,requires_grad=True);target=torch.zeros_like(pred)
        target[:,:,0,0]=1;region=torch.zeros(1,1,4,4);region[:,:,0,0]=1
        loss=balanced_region_l1(pred,target,region)
        torch.testing.assert_close(loss,torch.tensor(.5));loss.backward()
        self.assertGreater(pred.grad.abs().sum().item(),0)
        torch.testing.assert_close(balanced_region_l1(pred.detach(),target,torch.ones_like(region)),
                                   (pred.detach()-target).abs().mean())

    def test_opt_in_multimodal_semantic_readout_sees_visual_and_final_states_without_padding(self):
        torch.manual_seed(15)
        adapter=QueryResampler(16,16,8,4,1)
        model=SpatialTurboHead(UNet(),VAE(),None,adapter,SpatialBridge(16,16,16,2),[1,2,3,4]).eval()
        model.instruction_span=(6,2);model.multimodal_semantic_readout=True
        h=torch.randn(2,12,16,requires_grad=True);valid=torch.ones(2,12,dtype=torch.bool);valid[1,10:]=False
        mask=torch.zeros_like(valid);mask[:,1:5]=True;mask[0,6:10]=True;mask[1,6:8]=True;mask[0,11]=True;mask[1,9]=True
        actual=model.semantic_conditions(h,valid)
        torch.testing.assert_close(actual,adapter(h,mask),atol=0,rtol=0)
        actual.square().mean().backward()
        self.assertGreater(float(h.grad[:,1:5].abs().sum()),0)
        self.assertGreater(float(h.grad[0,11].abs().sum()),0)
        self.assertEqual(float(h.grad[1,10:].abs().sum()),0)


if __name__=='__main__':unittest.main()
