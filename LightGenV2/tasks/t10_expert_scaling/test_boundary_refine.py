import json
from pathlib import Path
import unittest
from .boundary_refine import pools,validate,lr,margin_and_protection,project_phase,metrics


class PolicyTests(unittest.TestCase):
    def setUp(self):self.c=json.loads((Path(__file__).parent/'configs/kather_n49_boundary_refine_20261027.json').read_text())
    def test_budget_and_checks(self):
        validate(self.c);self.assertEqual(12*512//16,384)
        for key,value in [('segments',3),('seed',27),('target_correct',600)]:
            with self.assertRaises(ValueError):validate(dict(self.c,**{key:value}))
    def test_stratified_pool_replay_and_views(self):
        import numpy as np
        y=np.repeat(np.arange(8),437);scores=np.random.default_rng(17).uniform(.01,1,(3496,8));scores/=scores.sum(1,keepdims=True)
        a=pools(y,scores);b=pools(y,scores)
        self.assertEqual(len(a),3)
        for (ix,hard),(other,_) in zip(a,b):
            np.testing.assert_array_equal(ix,other);self.assertEqual(len(set(ix)),256);self.assertEqual(np.bincount(y[ix]).tolist(),[32]*8)
            self.assertEqual(np.bincount(y[ix][hard]).tolist(),[8]*8);self.assertEqual(int(hard.sum()),64)
    def test_learning_rate_window(self):
        self.assertAlmostEqual(lr(1),5e-5);self.assertAlmostEqual(lr(2),1e-4);self.assertAlmostEqual(lr(12),2e-5)
        self.assertTrue(all(lr(n)>lr(n+1) for n in range(2,12)))
    def test_partial_identity_probe_has_finite_nll(self):
        import numpy as np
        p=np.full((2,8),.125);m=metrics(p,np.array([0,0]))
        self.assertTrue(np.isfinite(m['macro_nll']));self.assertEqual(sum(map(sum,m['confusion_matrix'])),2)


class LossTests(unittest.TestCase):
    def setUp(self):self.c=json.loads((Path(__file__).parent/'configs/kather_n49_boundary_refine_20261027.json').read_text())
    def test_easy_margin_supervision_stops_but_errors_move(self):
        import torch
        logits=torch.tensor([[3.,0.],[0.,1.]],requires_grad=True);p=logits.softmax(1);y=torch.tensor([0,0]);v,kl,easy=margin_and_protection(p,y,p.detach(),self.c)
        v.square().mean().backward();self.assertEqual(float(logits.grad[0].abs().sum()),0.);self.assertGreater(float(logits.grad[1].abs().sum()),0.)
        self.assertEqual(easy.tolist(),[True,False]);self.assertAlmostEqual(float(kl),0.,places=6)
    def test_circular_projection(self):
        import torch
        a=torch.tensor([3.13,-3.13]);p=torch.nn.Parameter(a+torch.tensor([.1,-.1]));rms=project_phase(p,a,.02)
        self.assertAlmostEqual(rms,.02,places=6)


if __name__=='__main__':unittest.main()
