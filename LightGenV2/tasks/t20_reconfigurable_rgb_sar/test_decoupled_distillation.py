import json,unittest
from pathlib import Path
import torch
from torch.nn import functional as F
from .decoupled_distillation import components,supervised_decoupled


class DecoupledTests(unittest.TestCase):
    def test_standard_KD_decomposition(self):
        torch.manual_seed(2);s=torch.randn(7,10,dtype=torch.float64);t=torch.randn_like(s);y=torch.arange(7);temp=4.
        a,b=components(s,t,y,temp);q=(t/temp).softmax(1).gather(1,y[:,None]).squeeze(1)
        full=F.kl_div((s/temp).log_softmax(1),(t/temp).softmax(1),reduction='none').sum(1)*temp**2
        torch.testing.assert_close(a+(1-q)*b,full,rtol=1e-10,atol=1e-12)

    def test_teacher_detached_and_wrong_rows_CE_only(self):
        s=torch.tensor([[1.,0.,0.],[1.,0.,0.]],dtype=torch.float64,requires_grad=True);t=torch.tensor([[5.,0.,1.],[5.,0.,1.]],dtype=torch.float64,requires_grad=True);y=torch.tensor([0,1]);v=supervised_decoupled(s,t,y,4.)
        torch.testing.assert_close(v[1],F.cross_entropy(s,y,label_smoothing=.02,reduction='none')[1]);v.mean().backward();self.assertIsNone(t.grad);self.assertTrue(torch.isfinite(s.grad).all())

    def test_identical_extreme_and_binary_are_finite(self):
        s=torch.tensor([[1000.,-1000.],[-1000.,1000.]],dtype=torch.float64,requires_grad=True);y=torch.tensor([0,1]);a,b=components(s,s.detach(),y,4.);self.assertLess(float((a+b).abs().max()),1e-12);supervised_decoupled(s,-s.detach(),y,4.).mean().backward();self.assertTrue(torch.isfinite(s.grad).all())

    def test_profile_has_no_TEST_fitting_or_new_inference(self):
        c=json.loads(Path(__file__).with_name('config_static_four_decoupled_readout_20261026.json').read_text());self.assertFalse(c['TEST_gradients']);self.assertFalse(c['TEST_teacher']);self.assertFalse(c['selection_uses_TEST']);self.assertEqual(c['new_inference_parameters'],0);self.assertLessEqual(c['fit_steps'],80)


if __name__=='__main__':unittest.main()
