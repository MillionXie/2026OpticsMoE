import unittest
import torch
from ..scene_source_condition_alignment import source_loss


class SourceVariation(unittest.TestCase):
    def test_exact_and_averaged_source(self):
        t=torch.arange(24,dtype=torch.float32).reshape(4,2,3)+100
        self.assertEqual(source_loss(t,t)[0].item(),0.)
        p=t.mean(0,keepdim=True).expand_as(t).clone().requires_grad_();loss,parts=source_loss(p,t)
        self.assertAlmostEqual(parts['source_delta'].item(),1.);loss.backward();self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(p.grad.abs().sum().item(),0.)
    def test_no_variation_or_wrong_group_rejected(self):
        with self.assertRaises(ValueError):source_loss(torch.zeros(4,2),torch.ones(4,2))
        with self.assertRaises(ValueError):source_loss(torch.zeros(3,2),torch.arange(6).reshape(3,2))


if __name__=='__main__':unittest.main()
