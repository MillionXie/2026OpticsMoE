import unittest,torch
from .continue_LG_C_optics import sharpness_backward,feature_retention_loss


class SharpnessTests(unittest.TestCase):
    def test_feature_teacher_zero_at_same_features_and_never_receives_gradient(self):
        target=torch.tensor([[.5,1.5],[.8,1.2]],requires_grad=True);current=target.detach().clone().requires_grad_();loss=feature_retention_loss(current,target)
        self.assertEqual(float(loss),0.);loss.backward();self.assertIsNone(target.grad)
        current=(target.detach()+.1).requires_grad_();loss=feature_retention_loss(current,target);self.assertGreater(float(loss),0);loss.backward();self.assertTrue(torch.isfinite(current.grad).all());self.assertIsNone(target.grad)
    def test_restores_parameters_but_keeps_perturbed_gradient(self):
        v=torch.nn.Parameter(torch.tensor([2.]));before=v.detach().clone();loss=v.square().sum()
        sharpness_backward(loss,[v],lambda:v.square().sum(),.05)
        torch.testing.assert_close(v,before);torch.testing.assert_close(v.grad,torch.tensor([4.1]))
    def test_failure_also_restores(self):
        v=torch.nn.Parameter(torch.tensor([2.]));before=v.detach().clone()
        with self.assertRaises(RuntimeError):sharpness_backward(v.square().sum(),[v],lambda:v.sum()*torch.tensor(float('nan')),.05)
        torch.testing.assert_close(v,before)


if __name__=='__main__':unittest.main()
