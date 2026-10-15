import unittest
import numpy as np
import torch
from lab_readout_20261016 import build_head,numpy_probability


class ReadoutTest(unittest.TestCase):
    def test_zero_residual_and_numpy_audit(self):
        torch.manual_seed(17);model=build_head();x=torch.randn(13,256)
        base=torch.rand(13,10);base=base/base.sum(1,keepdim=True)
        with torch.no_grad():p=(base.log()+model(x)).softmax(1)
        torch.testing.assert_close(p,base,atol=1e-7,rtol=1e-6)
        with torch.no_grad():model[-1].weight.normal_(0,.02);model[-1].bias.normal_(0,.02)
        with torch.no_grad():expected=(base.log()+model(x)).softmax(1).numpy()
        actual=numpy_probability(model.state_dict(),x.numpy(),base.numpy())
        np.testing.assert_allclose(actual,expected,atol=2e-7,rtol=2e-6)
    def test_only_readout_updates(self):
        torch.manual_seed(17);model=build_head();initial={k:v.clone() for k,v in model.state_dict().items()}
        x=torch.randn(16,256);labels=torch.arange(16)%10;opt=torch.optim.AdamW(model.parameters(),lr=.001)
        for _ in range(3):
            opt.zero_grad();torch.nn.functional.cross_entropy(model(x),labels).backward();opt.step()
        self.assertEqual(sum(p.numel() for p in model.parameters()),8554)
        self.assertTrue(all(not torch.equal(v,initial[k]) for k,v in model.state_dict().items()))


if __name__=='__main__':unittest.main()
