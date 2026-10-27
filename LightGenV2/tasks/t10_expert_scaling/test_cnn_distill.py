import unittest
import numpy as np
from .cnn_teacher_cache import admitted,codes_for_epoch


class PolicyTests(unittest.TestCase):
    def test_admission_requires_complement_val_gain_and_difficult_class_retention(self):
        self.assertTrue(admitted(128,[32]*4+[0]*4,626,588,[80]*8,{'1':58,'2':68,'4':60}))
        self.assertFalse(admitted(127,[32]*4+[0]*4,626,588,[80]*8,{'1':58}))
        self.assertFalse(admitted(128,[32]*4+[0]*4,625,588,[80]*8,{'1':58}))
        self.assertFalse(admitted(128,[32]*4+[0]*4,700,588,[80,80,67,80,80,80,80,80],{'2':68}))

    def test_eight_teacher_epochs_expose_each_train_patch_to_each_d4_once(self):
        codes=np.stack([codes_for_epoch(3496,e) for e in range(1,9)])
        np.testing.assert_array_equal(np.sort(codes,axis=0),np.broadcast_to(np.arange(8)[:,None],(8,3496)))


class TemperatureLossTests(unittest.TestCase):
    def test_t2_logits_and_probability_student_kl_matches_distribution_reference_and_microbatch(self):
        import torch
        from torch.distributions import Categorical,kl_divergence
        from .cnn_distill import temperature_kl_part
        logits=torch.tensor([[4.,1.],[1.,3.],[2.,0.],[1.,2.]])
        p=torch.tensor([[.7,.3],[.4,.6],[.8,.2],[.3,.7]],requires_grad=True);e=torch.tensor([True,True,False,True])
        expected=(kl_divergence(Categorical(logits=logits/2),Categorical(logits=p.log()/2))*e).sum()*4/3
        got=temperature_kl_part(p,logits,e,3);torch.testing.assert_close(got,expected);got.backward();g=p.grad.clone();p.grad.zero_()
        for start in [0,2]:temperature_kl_part(p[start:start+2],logits[start:start+2],e[start:start+2],3).backward()
        torch.testing.assert_close(p.grad,g)


if __name__=='__main__':unittest.main()
