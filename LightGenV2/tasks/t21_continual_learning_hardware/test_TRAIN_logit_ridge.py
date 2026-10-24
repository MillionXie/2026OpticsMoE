import unittest
import torch
from .distill_measured_TRAIN_logits import ridge_logits
from .refine_measured_ce import contrast_transform


class LogitRidgeTests(unittest.TestCase):
    def test_teacher_common_logit_does_not_change_fit(self):
        torch.manual_seed(4);z=torch.randn(60,8,dtype=torch.float64);z[:,-1]=1
        t=torch.randn(60,10,dtype=torch.float64);shift=torch.randn(60,1,dtype=torch.float64)
        self.assertTrue(torch.allclose(ridge_logits(z,t,.1),ridge_logits(z,t+shift,.1),atol=1e-12,rtol=1e-12))

    def test_transform_folds_into_original_linear(self):
        torch.manual_seed(5);x=1+.01*torch.randn(70,12,dtype=torch.float64);x=x/x.mean(1,keepdim=True)
        target=torch.randn(70,10,dtype=torch.float64);transform=contrast_transform(x)
        w=ridge_logits(x@transform.T,target,.01)
        self.assertEqual(tuple((w@transform).shape),(10,12))
        self.assertTrue(torch.allclose((x@transform.T)@w.T,x@(w@transform).T,atol=1e-10,rtol=1e-10))

    def test_invalid_strength_rejected(self):
        with self.assertRaises(ValueError):ridge_logits(torch.ones(3,2,dtype=torch.float64),torch.ones(3,2,dtype=torch.float64),0)


if __name__=='__main__':unittest.main()
