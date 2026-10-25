import unittest,json
from pathlib import Path
import numpy as np
import torch
from .refine_unified import conditioned_inputs,domain_balance,validate_profile


class RefinementTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)
    def test_conditioning_folds_into_same_no_bias_linear(self):
        rng=np.random.default_rng(42);x=rng.normal(size=(90,20));v=rng.normal(size=(25,20));w=torch.as_tensor(rng.normal(size=(10,20)))
        z,zv,t=conditioned_inputs(x,v,.01);p=torch.linalg.solve(t,w.T).T
        torch.testing.assert_close(zv@p.T,torch.as_tensor(v)@w.T,atol=1e-10,rtol=1e-10)
        torch.testing.assert_close(p@t.T,w,atol=1e-10,rtol=1e-10)
    def test_conditioner_uses_only_TRAIN_covariance(self):
        x=np.random.default_rng(42).normal(size=(30,10));v=np.ones((4,10));_,_,a=conditioned_inputs(x,v,.01);_,_,b=conditioned_inputs(x,v*100,.01)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    def test_domain_collapse_not_hidden_by_global_balance(self):
        q=torch.tensor([[1.,0.],[1.,0.],[0.,1.],[0.,1.]],requires_grad=True);d=torch.tensor([0,0,1,1]);indices=torch.tensor([0,1])
        self.assertEqual(float((q.mean(0)-.5).square().sum()),0.);loss=domain_balance(q,d,indices);self.assertGreater(float(loss),.4);loss.backward();self.assertGreater(float(q.grad.norm()),0.)
    def test_profile_and_no_alternate_inference(self):
        cfg=json.loads(Path(__file__).with_name('config_unified_refine_20261025.json').read_text());validate_profile(cfg)
        with self.assertRaises(ValueError):validate_profile(dict(cfg,TEST_gradients=True))
        code=Path(__file__).with_name('refine_unified.py').read_text();self.assertIn("locked_before_TEST.json",code);self.assertNotIn('nn.Linear(',code);self.assertNotIn("model.shared_head.bias",code)


if __name__=='__main__':unittest.main()
