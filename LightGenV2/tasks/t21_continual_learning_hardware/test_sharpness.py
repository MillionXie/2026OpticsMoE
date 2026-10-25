import unittest,torch
from .continue_LG_C_optics import sharpness_backward


class SharpnessTests(unittest.TestCase):
    def test_restores_parameters_but_keeps_perturbed_gradient(self):
        v=torch.nn.Parameter(torch.tensor([2.]));before=v.detach().clone();loss=v.square().sum()
        sharpness_backward(loss,[v],lambda:v.square().sum(),.05)
        torch.testing.assert_close(v,before);torch.testing.assert_close(v.grad,torch.tensor([4.1]))
    def test_failure_also_restores(self):
        v=torch.nn.Parameter(torch.tensor([2.]));before=v.detach().clone()
        with self.assertRaises(RuntimeError):sharpness_backward(v.square().sum(),[v],lambda:v.sum()*torch.tensor(float('nan')),.05)
        torch.testing.assert_close(v,before)


if __name__=='__main__':unittest.main()
