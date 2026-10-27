import unittest
import torch
from ..qwen_vl_semantic_loss import CaptionImageLoss


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.randn(3,8))
    def encode_image(self,image):
        return image.mean((2,3))@self.weight


class CaptionLossTests(unittest.TestCase):
    def test_frozen_teacher_keeps_image_gradient_without_any_gt_image(self):
        model=CaptionImageLoss(Encoder());image=torch.zeros(2,3,256,256,requires_grad=True)
        loss=model(image,torch.randn(2,8));loss.backward()
        self.assertTrue(torch.isfinite(image.grad).all());self.assertGreater(float(image.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.encoder.parameters()))


if __name__=='__main__':
    unittest.main()
