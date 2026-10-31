import unittest
import torch
from ..scene_counterfactual_condition_alignment import correction_loss


class CounterfactualConditions(unittest.TestCase):
    def test_exact_and_collapsed_gradients(self):
        target={k:torch.tensor([[1.,2.],[2.,1.],[3.,4.]])for k in ['text','pooled']}
        exact,_=correction_loss(target,target);self.assertEqual(exact.item(),0.)
        pred={k:torch.zeros_like(v,requires_grad=True)for k,v in target.items()};loss,parts=correction_loss(pred,target);self.assertAlmostEqual(parts['text_counterfactual'].item(),1.);loss.backward()
        self.assertTrue(all(torch.isfinite(v.grad).all()and v.grad.abs().sum()>0 for v in pred.values()))

    def test_common_offset_does_not_hide_conditional_failure(self):
        t=torch.tensor([[1.,0.],[0.,1.],[-1.,-1.]])+100
        target=dict(text=t,pooled=t);pred={k:v.mean(0,keepdim=True).expand_as(v)for k,v in target.items()};_,parts=correction_loss(pred,target)
        self.assertLess(parts['text_absolute'].item(),.001);self.assertAlmostEqual(parts['text_counterfactual'].item(),1.)
        with self.assertRaises(ValueError):correction_loss({k:torch.ones(3,2)for k in target},{k:torch.ones(3,2)for k in target})


if __name__=='__main__':unittest.main()
