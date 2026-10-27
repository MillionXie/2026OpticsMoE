import json
from pathlib import Path
import unittest
import numpy as np
from .mixup_curriculum import fraction,learning_rate,sample_indices,validate

class PilotTests(unittest.TestCase):
    def setUp(self):self.c=json.loads((Path(__file__).parent/'configs/kather_n49_mixup_curriculum_20261027.json').read_text())
    def test_endpoint_and_adequate_full49_budget(self):
        values=[fraction('power_curriculum_mixup',i,self.c) for i in range(1,33)]
        self.assertEqual(values[0],0.)
        self.assertEqual(values[15],1.)
        self.assertTrue(all(a<=b for a,b in zip(values,values[1:])))
        self.assertEqual(sum(v==1 for v in values),17)
        self.assertTrue(all(fraction('mixup',i,self.c)==1 for i in range(1,33)))
    def test_balanced_rotating_coverage(self):
        y=np.repeat(np.arange(8),437)
        selections=[sample_indices(y,i,17) for i in range(1,5)]
        for ix in selections:
            self.assertEqual(len(np.unique(ix)),1024)
            self.assertEqual(np.bincount(y[ix],minlength=8).tolist(),[128]*8)
        self.assertEqual(len(set(np.concatenate(selections))),3496)
    def test_warmup_and_decay(self):
        rates=[learning_rate(i,self.c) for i in range(1,33)]
        self.assertAlmostEqual(rates[0],.0001)
        self.assertAlmostEqual(rates[2],.0003)
        self.assertAlmostEqual(rates[-1],.00003)
        self.assertTrue(all(a>b for a,b in zip(rates[2:],rates[3:])))
    def test_target_and_resource_constraints(self):
        validate(self.c)
        self.assertGreaterEqual(616/752,.809+.01)
        cfg=dict(self.c,seed=27)
        with self.assertRaises(ValueError):validate(cfg)

if __name__=='__main__':unittest.main()
