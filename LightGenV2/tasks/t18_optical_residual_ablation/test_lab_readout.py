import unittest
import numpy as np
import torch
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tempfile
from lab_readout_20261016 import build_head,numpy_probability,initialized_head,require_simulation_handoff,sha


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
    def test_simulation_weights_and_buffers_loaded_exactly(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent/'runs') as directory:
            model=build_head()
            with torch.no_grad():model[-1].weight.fill_(.123)
            mean=np.arange(256,dtype=np.float32)/256;std=np.ones(256,dtype=np.float32)*.2
            path=Path(directory)/'head.pt';torch.save(dict(head=model.state_dict(),mean=mean,std=std),path)
            entry=dict(checkpoint=str(path),sha256=sha(path),optical_checkpoint_sha256='frozen',model_key='head',mean_key='mean',std_key='std')
            actual,m,s=initialized_head(entry,'frozen')
            for key,v in model.state_dict().items():self.assertTrue(torch.equal(actual.state_dict()[key],v))
            np.testing.assert_array_equal(m,mean);np.testing.assert_array_equal(s,std)
            with self.assertRaises(AssertionError):initialized_head(entry,'wrong optical parent')
    def test_missing_simulation_head_never_trains_random_fallback(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent/'runs') as directory:
            root=Path(directory);args=SimpleNamespace(train=root/'train',out=root/'readout')
            with patch('lab_readout_20261016.time.monotonic',side_effect=[0,3001]):
                with self.assertRaises(TimeoutError):require_simulation_handoff(args)
            self.assertFalse(args.out.exists())


if __name__=='__main__':unittest.main()
