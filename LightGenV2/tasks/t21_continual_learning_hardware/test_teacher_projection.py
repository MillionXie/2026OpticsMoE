import unittest
import torch
from .distill_offline_teacher_projection import solve_projection


class ProjectionTests(unittest.TestCase):
    def test_exact_linear_teacher(self):
        torch.manual_seed(5);x=torch.randn(200,12,dtype=torch.float64);x[:,-1]=1;w=torch.randn(10,12,dtype=torch.float64)
        fitted=solve_projection(x.T@x/200,x.T@(x@w.T)/200,1e-12)
        torch.testing.assert_close(fitted,w,atol=1e-9,rtol=1e-9)
    def test_intercept_not_penalized(self):
        x=torch.eye(4,dtype=torch.float64);y=torch.zeros(4,2,dtype=torch.float64);y[-1]=3
        w=solve_projection(x,y,100);torch.testing.assert_close(w[:,-1],torch.full((2,),3.,dtype=torch.float64))


if __name__=='__main__':unittest.main()
