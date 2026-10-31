import unittest
import torch
from ..scene_condition_gap import compare


class ConditionAudit(unittest.TestCase):
    def test_long_vector_self_identity_uses_precise_reduction(self):
        torch.manual_seed(1042);x=torch.randn(1,77,2048)*torch.logspace(-3,3,2048)
        result=compare(x,x)
        self.assertEqual(result['relative_RMSE'],0)
        self.assertLess(abs(result['cosine']-1),1e-10)
    def test_known_scale_difference(self):
        x=torch.tensor([1.,2.,3.]);r=compare(2*x,x)
        self.assertAlmostEqual(r['relative_RMSE'],1,12)
        self.assertAlmostEqual(r['cosine'],1,12)
    def test_invalid_shapes_and_values_rejected(self):
        with self.assertRaises(ValueError):compare(torch.ones(3),torch.ones(4))
        with self.assertRaises(ValueError):compare(torch.tensor([float('nan')]),torch.ones(1))


if __name__=='__main__':unittest.main()
