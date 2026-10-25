import json,unittest
from pathlib import Path
import numpy as np
import torch
from .relational_distillation import relations,relational_losses,teacher_representations
from .augmented_teacher import ImageTeacher,teacher_logits
from .train_shared_frontend import validate


class RelationalTests(unittest.TestCase):
    def test_teacher_embedding_exact_forward_and_no_grad(self):
        torch.set_num_threads(2);torch.manual_seed(3);m=ImageTeacher().eval()
        d=dict(images=np.random.default_rng(4).integers(0,256,(4,56,56,3),dtype=np.uint8),labels=np.arange(4))
        logits,emb=teacher_representations(m,d,'cpu',2)
        torch.testing.assert_close(logits,teacher_logits(m,d,'cpu',2),atol=0,rtol=0)
        self.assertEqual(emb.shape,(4,256));self.assertFalse(emb.requires_grad)

    def test_translation_rotation_scale_invariant(self):
        torch.manual_seed(4);s=torch.randn(8,6);q,_=torch.linalg.qr(torch.randn(6,6));t=3*(s@q)+7
        a,b=relational_losses(s,t,torch.tensor([0]*4+[1]*4))
        self.assertLess(float(a+b),1e-10)

    def test_finite_student_only_gradient_and_different_dimensions(self):
        s=torch.randn(8,784,requires_grad=True);t=torch.randn(8,256,requires_grad=True)
        a,b=relational_losses(s,t,torch.tensor([0]*4+[1]*4));(2*a+4*b).backward()
        self.assertIsNone(t.grad);self.assertTrue(torch.isfinite(s.grad).all());self.assertGreater(float(s.grad.norm()),0)
        z=torch.zeros(8,6,requires_grad=True);a,b=relational_losses(z,t,torch.tensor([0]*4+[1]*4));(a+b).backward();self.assertTrue(torch.isfinite(z.grad).all())

    def test_small_domain_and_fixed_profile(self):
        x=torch.randn(2,4,requires_grad=True);a,b=relational_losses(x,torch.randn(2,8),torch.tensor([0,1]));(a+b).backward();self.assertTrue(torch.isfinite(x.grad).all())
        c=json.loads(Path(__file__).with_name('config_static_four_relational_KD_20261026.json').read_text());validate(c)
        self.assertFalse(c['recalibrate_gains']);self.assertFalse(c['TEST_teacher']);self.assertFalse(c['TEST_gradients']);self.assertEqual(c['readout_fit_steps'],0)


if __name__=='__main__':unittest.main()
