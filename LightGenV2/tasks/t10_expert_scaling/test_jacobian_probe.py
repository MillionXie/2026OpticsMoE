import unittest
import numpy as np
from .jacobian_probe import solve_gram,competitors,stats,probe_output


class PolicyTests(unittest.TestCase):
    def test_weighted_dual_solution_matches_primal_without_row_unit_scaling(self):
        J=np.array([[1.,.2,0.],[.1,1.,.3],[.01,0.,.002]])
        w=np.array([.5,.5,1.]);b=np.array([.2,.3,0.]);A=np.sqrt(w)[:,None]*J;t=np.sqrt(w)*b;K=A@A.T
        alpha,diag=solve_gram(K,t);lam=diag['ridge_lambda'];dual=A.T@alpha
        primal=np.linalg.solve(A.T@A+lam*np.eye(3),A.T@t)
        np.testing.assert_allclose(dual,primal,atol=1e-12);self.assertAlmostEqual(lam,.01*np.trace(K)/3)
        self.assertTrue(0<=diag['target_energy_below_lambda_fraction']<=1)
        with self.assertRaises(ValueError):solve_gram(np.zeros((2,2)),np.ones(2))

    def test_fixed_complex_competitor_cannot_hide_full_argmax_error(self):
        parent=np.full((2,8),.01);parent[:,2]=.6;parent[:,1]=.3;y=np.array([2,2]);fixed=competitors(parent,y)
        self.assertEqual(fixed.tolist(),[1,1])
        trial=parent.copy();trial[0,0]=.8;trial[0,1]=.1
        result,_=stats(trial,y,np.array(['guard_c','guard_c']),parent,fixed)
        self.assertEqual(result['guard_c_clean']['new_wrong'],1);self.assertGreater(result['guard_c_clean']['fixed_margin_change'],0)
        self.assertEqual(result['guard_c_clean']['competitor_switches'],1)

    def test_S_D_competitor_stays_pairwise_even_if_another_class_larger(self):
        parent=np.array([[.01,.3,.6,.01,.05,.01,.01,.01],[.01,.05,.6,.01,.3,.01,.01,.01]])
        self.assertEqual(competitors(parent,np.array([1,4])).tolist(),[4,1])

    def test_correct_guard_can_protect_actual_nearest_without_changing_fit_pair(self):
        p=np.full((2,8),.01);p[:,1]=.6;p[:,2]=.3;p[:,4]=.05;y=np.array([1,1]);tags=np.array(['fit_s','guard_s'])
        self.assertEqual(competitors(p,y,tags,True).tolist(),[4,2])
        self.assertEqual(competitors(p,y).tolist(),[4,4])


class TorchTests(unittest.TestCase):
    def test_router_uses_full_model_not_frozen_router_field(self):
        import torch
        class Tiny:
            router_phase=torch.nn.Parameter(torch.tensor(2.))
            def __call__(self,rgb):return dict(probabilities=rgb*self.router_phase)
        m=Tiny();p=probe_output(m,torch.tensor([3.]),None,'router_phase')['probabilities'];p.sum().backward()
        self.assertEqual(float(p),6.);self.assertEqual(float(m.router_phase.grad),3.)
        with self.assertRaises(ValueError):probe_output(m,torch.tensor([3.]),None,'expert_phases.0')


if __name__=='__main__':unittest.main()
