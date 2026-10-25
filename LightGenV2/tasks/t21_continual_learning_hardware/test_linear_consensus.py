import unittest,torch
from .distill_linear_consensus import equal_consensus,D_development_bands
from .distill_margin_teacher import whole_stage_band_key


class LinearConsensusTests(unittest.TestCase):
    def test_D_requires_all_four_tasks_both_bounds(self):
        bands=D_development_bands();scores={n:dict(n=1000,correct=lo) for n,(lo,hi) in bands.items()}
        self.assertEqual(whole_stage_band_key(scores,bands),(0,0))
        scores['physical_binary_raw']['correct']=634;self.assertEqual(whole_stage_band_key(scores,bands),(-1,-1))
    def test_fixed_equal_entire_head_function_and_no_teacher_gradient(self):
        q=torch.randn(7,8,dtype=torch.float64);a=torch.randn(10,8,requires_grad=True);b=torch.randn(10,8,requires_grad=True)
        z=equal_consensus(q,[a,b]);expected=q@((a.detach().double()+b.detach().double())/2).T;expected-=expected.mean(1,keepdim=True)
        torch.testing.assert_close(z,expected);self.assertFalse(z.requires_grad)
        with self.assertRaises(ValueError):equal_consensus(q,[a])


if __name__=='__main__':unittest.main()
