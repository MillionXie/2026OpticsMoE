import unittest
from types import SimpleNamespace as NS
import torch
from .robust_holdout import split_train,paired_loss,score

class Tests(unittest.TestCase):
    def test_split(self):
        rows=[dict(split='train',product_id=str(i),sample_id=f'{i}_{j}') for i in range(200) for j in range(8)]
        groups=dict(train=rows,gallery=[],query=[dict(sample_id='test')])
        fit,val,audit=split_train(groups)
        self.assertEqual(len(fit['train']),1200);self.assertEqual(len(val['query']),400)
        self.assertFalse(set(audit['fitting_ids'])&set(audit['validation_ids']))
        self.assertEqual(split_train(groups)[2],audit)
    def test_paired_gradient(self):
        torch.manual_seed(1);z=torch.randn(4,64,requires_grad=True);clean=torch.randn(4,64);bank=torch.randn(8,64)
        logits=torch.randn(4,4,requires_grad=True);p=logits.softmax(-1)
        model=NS(vision=NS(optics=NS(router=NS(last=dict(probabilities=p)))),language=NS(optics=NS(router=NS(last=dict(probabilities=p)))))
        routes={name:torch.randn(4,4).softmax(-1) for name in ('vision','language')}
        x,y=paired_loss(z,clean,bank,None,model,routes,2);(x+y).backward()
        self.assertTrue(torch.isfinite(z.grad).all());self.assertGreater(float(logits.grad.abs().sum()),0)
    def test_clean_guard(self):
        a=dict(test=dict(hit_at_1=.79,map_at_10=.8),validation_noisy=dict(hit_at_1=.9))
        b=dict(test=dict(hit_at_1=.81,map_at_10=.8),validation_noisy=dict(hit_at_1=.7))
        self.assertGreater(score(b,.8,True),score(a,.8,True))

if __name__=='__main__':unittest.main()
