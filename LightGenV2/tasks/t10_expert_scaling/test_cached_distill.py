import unittest
from .cached_distill import calibrated_weight,calibration_indices,conditional_kl_part
from .reheat_refine import rate


class PolicyTests(unittest.TestCase):
    def test_twenty_round_stable_schedule_and_budget(self):
        c=dict(schedule_type='distill',rounds=20,peak_lr=.0003,end_lr=.00003)
        self.assertAlmostEqual(rate(1,c),.0001);self.assertAlmostEqual(rate(2,c),.0002)
        self.assertAlmostEqual(rate(3,c),.0003);self.assertAlmostEqual(rate(20,c),.00003)
        self.assertEqual(20*64,1280);self.assertEqual(20*1024,20480)

    def test_train_gradient_ratio_is_calibrated_and_bounded(self):
        value,ratio=calibrated_weight([2.,2.],[1.,1.]);self.assertEqual(value,1.);self.assertEqual(ratio,.5)
        self.assertEqual(calibrated_weight([100.],[1.])[0],3.)
        with self.assertRaises(ValueError):calibrated_weight([1.],[0.])

    def test_calibration_batches_are_disjoint_and_class_balanced(self):
        import numpy as np
        y=np.repeat(np.arange(8),20);q=np.eye(8)[y]*.9+.1/8;p=q.copy();p[::2]=np.eye(8)[(y[::2]+1)%8]*.9+.1/8
        a,b=calibration_indices(y,q,p);self.assertEqual(len(set(a)|set(b)),32)
        self.assertEqual(np.bincount(y[a]).tolist(),[2]*8);self.assertEqual(np.bincount(y[b]).tolist(),[2]*8)


class ConditionalLossTests(unittest.TestCase):
    def test_whole_effective_batch_denominator_matches_microbatch_gradients(self):
        import torch
        q=torch.tensor([[.8,.2],[.7,.3],[.2,.8],[.4,.6]])
        eligible=torch.tensor([True,True,False,True]);p=torch.tensor([[.6,.4],[.6,.4],[.3,.7],[.5,.5]],requires_grad=True)
        loss=conditional_kl_part(p,q,eligible,3);loss.backward();reference=p.grad.clone();p.grad.zero_()
        for start in [0,2]:conditional_kl_part(p[start:start+2],q[start:start+2],eligible[start:start+2],3).backward()
        torch.testing.assert_close(reference,p.grad)
        zero=conditional_kl_part(p,q,torch.zeros(4,dtype=torch.bool),0);self.assertEqual(float(zero),0.)


if __name__=='__main__':unittest.main()
