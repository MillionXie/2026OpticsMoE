import unittest
import torch
from torch import nn
from ..scene_perceptual_rollout_recovery import perceptual_agreement


class FeatureMetric(nn.Module):
    def __init__(self):
        super().__init__();self.feature=nn.Conv2d(3,4,3,padding=1)
    def forward(self,a,b):return (self.feature(a)-self.feature(b)).square().mean((1,2,3))


class PerceptualContract(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1042);self.metric=FeatureMetric().eval().requires_grad_(False)
    def test_student_gradient_without_teacher_or_metric_gradient(self):
        student=torch.rand(1,3,32,32,requires_grad=True);teacher=torch.rand_like(student,requires_grad=True)
        loss=perceptual_agreement(self.metric,student,teacher);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertGreater(float(student.grad.abs().sum()),0)
        self.assertIsNone(teacher.grad);self.assertTrue(all(p.grad is None for p in self.metric.parameters()))
    def test_identical_images_have_zero_loss(self):
        image=torch.rand(1,3,16,16)
        self.assertEqual(float(perceptual_agreement(self.metric,image,image)),0)
    def test_mismatched_images_rejected(self):
        with self.assertRaises(ValueError):perceptual_agreement(self.metric,torch.zeros(1,3,16,16),torch.zeros(1,3,17,16))
    def test_non_rgb_rejected(self):
        with self.assertRaises(ValueError):perceptual_agreement(self.metric,torch.zeros(1,4,16,16),torch.zeros(1,4,16,16))
    def test_trainable_metric_rejected(self):
        with self.assertRaises(ValueError):perceptual_agreement(self.metric.requires_grad_(True),torch.zeros(1,3,16,16),torch.zeros(1,3,16,16))
    def test_unclamped_decoder_values_have_finite_gradient(self):
        student=torch.full((1,3,16,16),1.2,requires_grad=True)
        loss=perceptual_agreement(self.metric,student,torch.zeros_like(student));loss.backward()
        self.assertTrue(torch.isfinite(student.grad).all());self.assertGreater(float(student.grad.abs().sum()),0)


if __name__=='__main__':unittest.main()
