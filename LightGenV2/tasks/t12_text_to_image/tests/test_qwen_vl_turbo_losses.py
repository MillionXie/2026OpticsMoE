import unittest
import torch
from torch import nn

from LightGenV2.tasks.t12_text_to_image.qwen_vl_turbo_losses import (
    FrozenCLIPSemantic,PatchDiscriminator,crop_regions,discriminator_views,
    hinge_discriminator_loss,region_boxes)


class FakeCLIP(nn.Module):
    def __init__(self):
        super().__init__();self.scale=nn.Parameter(torch.tensor(2.))
    def encode_image(self,x):return x.mean((2,3))*self.scale


class LossTests(unittest.TestCase):
    def test_foreground_losses_are_area_normalized_and_have_boundary_gradients(self):
        from ..qwen_vl_turbo_losses import foreground_detail_losses
        target=torch.zeros(2,3,32,32);alpha=torch.zeros(2,1,32,32)
        alpha[0,:,8:24,8:24]=1;alpha[1,:,12:20,12:20]=1
        prediction=alpha.expand(-1,3,-1,-1).clone().requires_grad_()
        pixel,edge=foreground_detail_losses(prediction,target,alpha)
        self.assertAlmostEqual(float(pixel.detach()),1.);self.assertGreater(float(edge.detach()),0.)
        (pixel+edge).backward();self.assertTrue(torch.isfinite(prediction.grad).all())
        zero=foreground_detail_losses(target,target,alpha)
        self.assertEqual([float(v) for v in zero],[0.,0.])
        with self.assertRaises(ValueError):foreground_detail_losses(target,target,alpha*0)

    def test_crop_has_image_gradient_and_empty_region_fallback(self):
        masks=torch.zeros(2,1,32,32);masks[0,:,8:24,6:20]=1
        boxes=region_boxes(masks)
        torch.testing.assert_close(boxes[1],torch.tensor([0.,0.,1.,1.]))
        x=torch.randn(2,3,32,32,requires_grad=True)
        cropped=crop_regions(x,boxes,16)
        self.assertEqual(tuple(cropped.shape),(2,3,16,16))
        cropped.square().mean().backward()
        self.assertGreater(x.grad.abs().sum().item(),0)

    def test_frozen_clip_still_trains_generator_pixels(self):
        model=FakeCLIP();loss=FrozenCLIPSemantic(model)
        x=torch.zeros(2,3,32,32,requires_grad=True)
        text=torch.tensor([[1.,0,0],[0,1.,0]])
        value=loss(x,text);value.backward()
        self.assertGreater(x.grad.abs().sum().item(),0)
        self.assertIsNone(model.scale.grad)
        self.assertFalse(model.scale.requires_grad)

    def test_critic_generator_and_discriminator_gradients_are_separate(self):
        critic=PatchDiscriminator(width=8).eval().requires_grad_(False)
        x=torch.randn(2,3,64,64,requires_grad=True)
        boxes=torch.tensor([[0.,0.,1.,1.]]*2)
        fake=discriminator_views(x,boxes)
        (-critic(fake).mean()).backward()
        self.assertGreater(x.grad.abs().sum().item(),0)
        self.assertTrue(all(p.grad is None for p in critic.parameters()))
        critic.train().requires_grad_(True)
        dloss=hinge_discriminator_loss(critic(torch.randn_like(fake)),critic(fake.detach()))
        dloss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in critic.parameters()))


if __name__=='__main__':unittest.main()
