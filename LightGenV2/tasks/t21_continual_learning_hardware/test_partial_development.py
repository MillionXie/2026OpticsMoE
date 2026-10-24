import unittest
import numpy as np
import torch
from .fit_partial_development import selected_ids,derivative_pool,DISCLOSURE


class PartialDevelopmentTests(unittest.TestCase):
    def test_same_source_admission_despite_capture_permutation(self):
        ids=np.arange(1000);y=np.repeat(np.arange(10),100);a=selected_ids(ids,y)
        q=np.random.default_rng(4).permutation(1000)
        self.assertEqual(a,selected_ids(ids[q],y[q]));self.assertEqual(len(a),100)
        self.assertEqual(np.bincount(y[a]).tolist(),[10]*10)

    def test_derivatives_not_original_EVAL_pool_and_disclosure(self):
        torch.manual_seed(2);x=1+.1*torch.randn(1000,12,dtype=torch.float64);x=x/x.mean(1,keepdim=True)
        y=torch.arange(10).repeat_interleave(100);e=x[::10].clone();ey=y[::10]
        a,labels=derivative_pool(x,y,e,ey)
        self.assertEqual(a.shape,(2000,12));self.assertEqual(len(labels),2000)
        self.assertFalse(torch.equal(a[:100],e));self.assertTrue(torch.allclose(a.mean(1),torch.ones(2000,dtype=torch.float64)))
        self.assertFalse(DISCLOSURE['valid_for_formal_test_table']);self.assertEqual(DISCLOSURE['independent_test_samples'],0)


if __name__=='__main__':unittest.main()
