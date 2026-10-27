import unittest
import numpy as np
from .boundary_transfer_probe import matched_groups,transfer_metrics,cached_suffix
from .phase_repair_probe import select_groups


class PolicyTests(unittest.TestCase):
    def test_new_fit_and_observation_are_adjacent_margin_pairs(self):
        y=np.repeat(np.arange(8),100);p=np.full((800,8),.001);p[np.arange(800),y]=.99
        for k,other in [(1,4),(4,1)]:
            for j,i in enumerate(range(k*100,k*100+48)):
                p[i,k]=.3+j*.001;p[i,other]=.4
        for k in [1,2,4]:
            p[k*100+60:k*100+68,k]=.45;p[k*100+60:k*100+68,0]=.42
            p[k*100+70:k*100+78,k]=.5;p[k*100+70:k*100+78,0]=.36
        groups,pairs=matched_groups(y,p);old=set(select_groups(y,p)['repair']);all_indices=sum(groups.values(),[])
        self.assertEqual(len(all_indices),len(set(all_indices)));self.assertTrue(set(all_indices).isdisjoint(old))
        for name in ['s','d']:
            self.assertEqual(len(groups['fit_'+name]),8);self.assertEqual(len(groups['observe_'+name]),8)
            for pair in pairs[name]:
                self.assertEqual(len(set(pair)&set(groups['fit_'+name])),1);self.assertEqual(len(set(pair)&set(groups['observe_'+name])),1)
        self.assertEqual(groups,matched_groups(y,p)[0])

    def test_actual_repairs_do_not_hide_pairwise_improvements(self):
        q=np.array([[.3,.4,.2,.1],[.3,.4,.2,.1]]);p=np.array([[.35,.3,.36,.0],[.4,.3,.2,.1]])
        m=transfer_metrics(p,np.array([0,0]),np.array(['fit_s','fit_s']),q)
        self.assertEqual(m['fit_s_clean']['corrected'],0);self.assertEqual(m['fit_s_d4']['corrected'],1)


class TorchTests(unittest.TestCase):
    def test_expert_suffix_keeps_gradient_through_real_oeo_and_fixed_global(self):
        import torch
        from unittest.mock import patch
        class Tiny:
            expert_phases=[None,torch.nn.Parameter(torch.tensor(2.))]
            def phase_step(self,field,phase,expert):
                self.called=expert;return (field.real*phase).to(torch.complex64)
        model=Tiny();field=torch.tensor([3.]);rp=torch.tensor([1.])
        with patch('LightGenV2.tasks.t10_expert_scaling.boundary_refine.suffix',side_effect=lambda m,f,r:dict(probabilities=f*4)):
            p=cached_suffix(model,field,rp,'expert_phases.1')['probabilities'];p.sum().backward()
        self.assertTrue(model.called);self.assertEqual(float(p),24.);self.assertEqual(float(model.expert_phases[1].grad),12.)
        with self.assertRaises(ValueError):cached_suffix(model,field,rp,'router_phase')


if __name__=='__main__':unittest.main()
