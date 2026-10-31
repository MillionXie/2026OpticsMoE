import unittest
from ..scene_condition_path_diagnostic import centered_stats


class CenteredConditions(unittest.TestCase):
    def test_exact_and_collapsed_condition_variation(self):
        import torch
        t=torch.tensor([[1.,0.],[0.,1.],[-1.,-1.]])
        exact=centered_stats(t,t);self.assertAlmostEqual(exact['variation_ratio'],1.);self.assertAlmostEqual(exact['centered_relative_MSE'],0.)
        collapsed=centered_stats(torch.ones_like(t),t);self.assertEqual(collapsed['variation_ratio'],0.);self.assertAlmostEqual(collapsed['centered_relative_MSE'],1.)
    def test_mean_offset_is_separated_from_instruction_variation(self):
        import torch
        t=torch.tensor([[1.,0.],[0.,1.],[-1.,-1.]])
        scaled=centered_stats(.1*t+10,t);self.assertAlmostEqual(scaled['variation_ratio'],.1,places=5);self.assertAlmostEqual(scaled['centered_relative_MSE'],.81,places=5)
        with self.assertRaises(ValueError):centered_stats(t,torch.zeros_like(t))


if __name__=='__main__':unittest.main()
