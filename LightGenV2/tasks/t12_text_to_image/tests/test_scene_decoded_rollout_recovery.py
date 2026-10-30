import unittest
from ..scene_decoded_rollout_recovery import image_agreement,decoded_agreement


class DecodedLossTest(unittest.TestCase):
    def test_identical_images_have_zero_pixel_and_edge_losses(self):
        import torch
        x=torch.randn(1,3,16,16);p,e=image_agreement(x,x)
        self.assertEqual(float(p),0);self.assertEqual(float(e),0)

    def test_decoder_transmits_gradient_to_student_but_never_teacher_or_decoder(self):
        import torch
        class Decoder(torch.nn.Module):
            dtype=torch.float32
            def __init__(self):super().__init__();self.map=torch.nn.Conv2d(4,3,1)
            def decode(self,x,return_dict):return(self.map(x),)
        decoder=Decoder().requires_grad_(False);student=torch.randn(1,4,16,16,requires_grad=True);teacher=torch.randn_like(student,requires_grad=True)
        p,e=decoded_agreement(decoder,student,teacher,.13,True);(p+e).backward()
        self.assertGreater(float(student.grad.abs().sum()),0);self.assertTrue(torch.isfinite(student.grad).all())
        self.assertIsNone(teacher.grad);self.assertTrue(all(p.grad is None for p in decoder.parameters()))

    def test_trainable_decoder_or_wrong_image_channels_are_rejected(self):
        import torch
        decoder=torch.nn.Linear(2,2)
        with self.assertRaisesRegex(ValueError,'Frozen'):decoded_agreement(decoder,torch.ones(2),torch.ones(2),.13)
        with self.assertRaisesRegex(ValueError,'B3HW'):image_agreement(torch.ones(1,4,8,8),torch.ones(1,4,8,8))

    def test_missing_low_contrast_boundary_has_finite_nonzero_relative_loss(self):
        import torch
        wanted=torch.zeros(1,3,16,16);wanted[...,8:,:]=.05
        pixel,edge=image_agreement(torch.zeros_like(wanted),wanted)
        self.assertGreater(float(pixel),0);self.assertGreater(float(edge),.1)
        flat_pixel,flat_edge=image_agreement(torch.ones_like(wanted),torch.zeros_like(wanted))
        self.assertEqual(float(flat_edge),0);self.assertTrue(torch.isfinite(flat_pixel+flat_edge))


if __name__=='__main__':unittest.main()
