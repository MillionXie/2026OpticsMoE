import json
from pathlib import Path
import unittest
from .sharpness_refine import validate, optimization_step, learning_rate, sample_indices


class StrategyTests(unittest.TestCase):
    def setUp(self):self.c=json.loads((Path(__file__).parent/'configs/kather_n49_sharpness_refine_20261027.json').read_text())
    def test_budget_and_resource_lock(self):
        validate(self.c)
        self.assertEqual(2*self.c['rounds']*1024,49152)
        for field,value in [('seed',27),('rho',.5),('rounds',3)]:
            with self.assertRaises(ValueError):validate(dict(self.c,**{field:value}))
    def test_paired_deterministic_sampling(self):
        import numpy as np
        y=np.repeat(np.arange(8),437)
        a=[sample_indices(y,n,17) for n in range(1,5)]
        self.assertEqual(len(set(np.concatenate(a))),3496)
        for n,ix in enumerate(a,1):
            np.testing.assert_array_equal(ix,sample_indices(y,n,17))
            self.assertEqual(np.bincount(y[ix]).tolist(),[128]*8)
    def test_finite_full_window_schedule(self):
        rates=[learning_rate(n,self.c) for n in range(1,25)]
        self.assertAlmostEqual(rates[0],1e-4);self.assertAlmostEqual(rates[2],3e-4);self.assertAlmostEqual(rates[-1],3e-5)


class OptimizerTests(unittest.TestCase):
    def test_sam_descent_gradient_and_restoration(self):
        import torch
        model=torch.nn.Linear(1,1,bias=False)
        with torch.no_grad():model.weight.fill_(1.)
        optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        seen=[]
        def closure():
            seen.append(float(model.weight.detach()))
            loss=model(torch.ones(1,1)).square().mean();loss.backward();return float(loss.detach())
        result=optimization_step(model,optimizer,closure,rho=.05,clip_norm=None)
        self.assertEqual(len(seen),2);self.assertAlmostEqual(seen[0],1.);self.assertAlmostEqual(seen[1],1.05,places=6)
        self.assertAlmostEqual(float(model.weight.detach()),.79,places=6)
        self.assertTrue(result['restored_before_optimizer']);self.assertAlmostEqual(result['rho_actual'],.05,places=6)
        self.assertGreater(result['sharpness_delta'],0)
    def test_exception_restores_unperturbed_weights(self):
        import torch
        model=torch.nn.Linear(1,1,bias=False)
        original=model.weight.detach().clone();optimizer=torch.optim.SGD(model.parameters(),lr=.1);calls=0
        def closure():
            nonlocal calls
            calls+=1
            if calls==2:raise RuntimeError('second pass failed')
            loss=model(torch.ones(1,1)).square().mean();loss.backward();return float(loss.detach())
        with self.assertRaisesRegex(RuntimeError,'second pass failed'):optimization_step(model,optimizer,closure,rho=.05)
        self.assertTrue(torch.equal(model.weight,original))


if __name__=='__main__':unittest.main()
