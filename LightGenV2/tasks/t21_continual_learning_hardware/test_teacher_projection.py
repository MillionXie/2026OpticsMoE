import unittest
import torch
from .distill_offline_teacher_projection import solve_projection,admit_legacy_partial,centered_jacobian_mean,solve_pair_projection,binary_source_pairs
import numpy as np
from .fit_augmented_residual_teacher import ResidualTeacher


class ProjectionTests(unittest.TestCase):
    def test_pair_projection_preserves_exact_linear_teacher(self):
        torch.manual_seed(4);x=torch.randn(200,8,dtype=torch.float64);x[:,-1]=1;teacher=torch.randn(10,8,dtype=torch.float64)
        delta=x[:100]-x[100:];margin=(teacher[1]-teacher[0])/2
        actual=solve_pair_projection(x.T@x/200,x.T@(x@teacher.T)/200,delta.T@delta/100,delta.T@(delta@margin)/100,1e-12)
        torch.testing.assert_close(actual,teacher,atol=1e-9,rtol=1e-9)
    def test_pair_projection_stationarity(self):
        torch.manual_seed(3);x=torch.randn(30,5,dtype=torch.float64);x[:,-1]=1;d=torch.randn(12,5,dtype=torch.float64);d[:,-1]=0
        y=torch.randn(30,10,dtype=torch.float64);target=torch.randn(12,dtype=torch.float64);strength=.02
        w=solve_pair_projection(x.T@x/30,x.T@y/30,d.T@d/12,d.T@target/12,strength).requires_grad_()
        loss=(x@w.T-y).square().mean(0).sum()/2+((d@((w[1]-w[0])/2)-target).square().mean())+strength*w[:,:-1].square().sum()/2
        loss.backward();self.assertLess(float(w.grad.abs().max()),1e-9)
    def test_source_pairs_complete_positive_first(self):
        pairs=binary_source_pairs(np.array([3,0,9,2,1]),torch.tensor([0,1,2,1,0]),np.array(['clevr','clevr','eurosat','clevr','clevr']))
        self.assertEqual(pairs.tolist(),[[1,4],[3,0]])
        with self.assertRaises(ValueError):binary_source_pairs(np.array([0]),torch.tensor([1]),np.array(['clevr']))
    def test_exact_GELU_teacher_Jacobian_against_autograd(self):
        torch.manual_seed(7);teacher=ResidualTeacher(torch.randn(10,6,dtype=torch.float64),hidden=4).double().eval()
        with torch.no_grad():teacher.residual[-1].weight.normal_()
        x=torch.randn(3,6,dtype=torch.float64);weight=torch.full((3,1),1/3,dtype=torch.float64)
        actual=centered_jacobian_mean(teacher,x,weight);expected=[]
        for row in x:
            expected.append(torch.autograd.functional.jacobian(lambda z:teacher(z[None])[0]-teacher(z[None])[0].mean(),row))
        expected=torch.stack(expected).mean(0);expected[:,-1]=0
        torch.testing.assert_close(actual,expected,atol=1e-10,rtol=1e-10)
    def test_exact_linear_teacher(self):
        torch.manual_seed(5);x=torch.randn(200,12,dtype=torch.float64);x[:,-1]=1;w=torch.randn(10,12,dtype=torch.float64)
        fitted=solve_projection(x.T@x/200,x.T@(x@w.T)/200,1e-12)
        torch.testing.assert_close(fitted,w,atol=1e-9,rtol=1e-9)
    def test_intercept_not_penalized(self):
        x=torch.eye(4,dtype=torch.float64);y=torch.zeros(4,2,dtype=torch.float64);y[-1]=3
        w=solve_projection(x,y,100);torch.testing.assert_close(w[:,-1],torch.full((2,),3.,dtype=torch.float64))
    def test_retired_teacher_rejected(self):
        with self.assertRaises(ValueError):admit_legacy_partial({'selection':{'schema':'all_TEST'}},{})
    def test_partial_disclosure_preserved(self):
        r=admit_legacy_partial(dict(selection={'schema':'T21_fixed_partial_development_100_of1000_v1'},EVAL_derivative_sources=100),dict(teacher_direct_original_EVAL_supervision=False,teacher_EVAL_derivative_gradient=True,valid_for_formal_test_table=False))
        self.assertTrue(r['student_inherits_EVAL_information']);self.assertEqual(r['independent_test_samples'],0)


if __name__=='__main__':unittest.main()
