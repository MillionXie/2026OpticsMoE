import unittest
import numpy as np
import torch
from .distill_measured_TRAIN import validate_teacher,objective,fit
from .refine_measured_ce import contrast_transform
from .recheck_TRAIN_temporal import drift_summary


class TRAINDistillationTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(73)

    def test_teacher_no_gradient(self):
        student=torch.randn(12,10,dtype=torch.float64,requires_grad=True)
        teacher=torch.randn(12,10,dtype=torch.float64,requires_grad=True)
        loss=objective(student,torch.arange(12)%10,teacher);loss.backward()
        self.assertTrue(torch.isfinite(student.grad).all());self.assertIsNone(teacher.grad)

    def test_TEST_or_wrong_identity_rejected(self):
        m=dict(passed=True,TEST_used=False,TEST_augmented=False,checkpoint_SHA256='abc',TRAIN_records=1000)
        ids=np.arange(1000);y=ids%10;validate_teacher(m,'abc',ids,y,ids,y)
        for bad in (dict(m,TEST_used=True),dict(m,TEST_augmented=True),dict(m,checkpoint_SHA256='other')):
            with self.assertRaises(ValueError):validate_teacher(bad,'abc',ids,y,ids,y)
        with self.assertRaises(ValueError):validate_teacher(m,'abc',ids,y,ids[::-1],y)

    def test_fixed_family_fold_and_finite_fit(self):
        x=torch.rand(60,8,dtype=torch.float64)+.2;x/=x.mean(1,keepdim=True)
        transform=contrast_transform(x);z=x@transform.T;teacher=torch.randn(60,10,dtype=torch.float64)
        w,loss=fit(z,torch.arange(60)%10,teacher,.1,8)
        self.assertEqual(tuple(w.shape),(10,8));self.assertTrue(np.isfinite(loss))
        torch.testing.assert_close(z@w.T,x@(w@transform).T,rtol=1e-10,atol=1e-10)

    def test_temporal_statistics_not_a_guard_override(self):
        old=np.ones(8);repeat=[old-.01,old,old+.01]
        steady=drift_summary(old,repeat);shifted=drift_summary(old,[r+.1 for r in repeat])
        self.assertAlmostEqual(steady['old_to_new_relative_L2'],0.)
        self.assertGreater(shifted['old_to_new_over_current_repeat'],10.)


if __name__=='__main__':unittest.main()
