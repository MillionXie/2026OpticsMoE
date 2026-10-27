import unittest
import numpy as np
from .phase_repair_probe import select_groups, grouped_metrics, phase_direction


class PolicyTests(unittest.TestCase):
    def test_near_confusions_exclude_extreme_and_wrong_pairs(self):
        y=np.repeat(np.arange(8),50);p=np.full((400,8),.01)
        p[np.arange(400),y]=.93
        p[50:55,1]=.4;p[50:55,4]=.5
        p[55,1]=.1;p[55,4]=.8
        p[56,1]=.4;p[56,2]=.5
        p[200:205,4]=.4;p[200:205,1]=.5
        groups=select_groups(y,p);self.assertEqual(groups['repair'],list(range(50,55))+list(range(200,205)))
        all_indices=sum(groups.values(),[]);self.assertEqual(len(all_indices),len(set(all_indices)))
        self.assertTrue(np.all(y[groups['complex']]==2))
        for name in ['replay','observe']:self.assertEqual(np.bincount(y[groups[name]]).tolist(),[4]*8)
        self.assertEqual(groups,select_groups(y,p))

    def test_no_extreme_fallback(self):
        y=np.repeat(np.arange(8),50);p=np.full((400,8),.01);p[np.arange(400),y]=.93
        self.assertEqual(select_groups(y,p)['repair'],[])

    def test_clean_and_d4_metrics_separate_repairs_from_harm(self):
        source=np.array([[.4,.6],[.4,.6],[.8,.2],[.8,.2]])
        current=np.array([[.6,.4],[.45,.55],[.4,.6],[.8,.2]])
        metrics,_=grouped_metrics(current,np.zeros(4,dtype=int),np.array(['repair','repair','complex','complex']),source)
        self.assertEqual(metrics['repair_clean']['corrected'],1)
        self.assertEqual(metrics['repair_d4']['corrected'],0)
        self.assertEqual(metrics['complex_clean']['new_wrong'],1)


class TorchTests(unittest.TestCase):
    def test_directions_are_unit_rms_and_descent(self):
        import torch
        g=torch.tensor([1e-4,-.1,0.,1.])
        for adam in [False,True]:
            d=phase_direction(g,adam);self.assertAlmostEqual(float(d.square().mean().sqrt()),1.,places=6)
            self.assertLess(float((d*g).sum()),0.)
        with self.assertRaises(ValueError):phase_direction(torch.zeros(2))


if __name__=='__main__':unittest.main()
