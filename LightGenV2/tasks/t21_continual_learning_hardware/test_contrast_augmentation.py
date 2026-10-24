import unittest
import torch
from .refine_measured_contrast import augment,contrast_objective,validate_profile
from .distill_measured_TRAIN import objective


class ContrastAugmentationTests(unittest.TestCase):
    def test_background_not_dropped(self):
        x=torch.ones(100,784,dtype=torch.float64)
        self.assertTrue(torch.equal(augment(x,x.mean(0,keepdim=True),torch.zeros(16,784,dtype=x.dtype)),x))

    def test_DC_restored_and_finite(self):
        torch.manual_seed(8);x=1+.01*torch.randn(100,784,dtype=torch.float64);x=x/x.mean(1,keepdim=True)
        noise=.005*torch.randn(16,784,dtype=x.dtype);noise-=noise.mean(1,keepdim=True)
        out=augment(x,x.mean(0,keepdim=True),noise)
        self.assertTrue(torch.isfinite(out).all());self.assertLess(float((out.mean(1)-1).abs().max()),1e-12)
        self.assertLess(float((out-x).std()),.01)

    def test_default_loss_exactly_unchanged(self):
        torch.manual_seed(2);a=torch.randn(8,10,dtype=torch.float64);b=torch.randn_like(a);y=torch.arange(8)
        self.assertTrue(torch.equal(contrast_objective(a,y,b),objective(a,y,b)))

    def test_wrong_teacher_can_be_excluded_without_label_change(self):
        a=torch.zeros(1,10,dtype=torch.float64);teacher=torch.zeros_like(a);teacher[0,1]=5;y=torch.zeros(1,dtype=torch.long)
        cfg=dict(CE_weight=0.,KD_weight=4.,focal_gamma=1.,wrong_teacher_KD_weight=0.)
        self.assertEqual(float(contrast_objective(a,y,teacher,cfg)),0.)
        cfg['wrong_teacher_KD_weight']=1.
        self.assertGreater(float(contrast_objective(a,y,teacher,cfg)),0.)

    def test_profile_budget_rejects_unknown_or_out_of_range(self):
        with self.assertRaises(ValueError):validate_profile(dict(schema='other',loss={}))
        with self.assertRaises(ValueError):validate_profile(dict(schema='T21_A_contrast_masked_KD_v1',loss=dict(CE_weight=1.,KD_weight=5.,focal_gamma=1.,wrong_teacher_KD_weight=.25)))


if __name__=='__main__':unittest.main()
