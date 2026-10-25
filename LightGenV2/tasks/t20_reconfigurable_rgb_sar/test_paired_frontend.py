import unittest
import numpy as np
import torch
from .diagnose_frontend_train import alignment,pair_batches


class PairedFrontendTests(unittest.TestCase):
    def test_pairs_never_cross_locations_or_labels(self):
        data=dict(ids=np.array([str(i)+s for i in range(8) for s in (':RGB',':SAR')]),domains=np.tile([0,1],8),labels=np.repeat(np.arange(8),2));b=list(pair_batches(data,4,42));self.assertEqual(sorted(np.concatenate(b).tolist()),list(range(16)))
        for ix in b:self.assertTrue(np.all(ix[::2]+1==ix[1::2]))
    def test_pair_alignment_preserves_discriminating_common_signal(self):
        x=torch.randn(4,128);f=x.repeat_interleave(2,0).requires_grad_();loss=alignment(f);self.assertLess(abs(float(loss)),1e-6);loss.backward();self.assertTrue(torch.isfinite(f.grad).all())


if __name__=='__main__':unittest.main()
