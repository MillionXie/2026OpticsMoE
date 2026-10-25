import unittest,numpy as np,torch
from .audit_student_transfer import linear_projection,comparison


class TransferAuditTests(unittest.TestCase):
    def test_linear_teacher_exactly_recoverable(self):
        torch.manual_seed(4);z=torch.randn(80,12,dtype=torch.float64);w=torch.randn(12,10,dtype=torch.float64);target=z@w;task=np.array(['a']*40+['b']*40)
        fitted,r=linear_projection(z,target,task);torch.testing.assert_close(z@fitted,target,rtol=1e-10,atol=1e-10);self.assertEqual(r['rank'],12)
    def test_equal_predictions_have_zero_KL_and_unit_agreement(self):
        x=torch.randn(20,10,dtype=torch.float64);m=comparison(x,x,x.argmax(1));self.assertEqual(m['agreement'],1);self.assertLess(abs(m['teacher_KL_T2']),1e-10);self.assertEqual(m['student_correct'],20)
    def test_nonlinear_teacher_not_assumed_representable(self):
        torch.manual_seed(7);z=torch.randn(100,3,dtype=torch.float64);target=torch.cat((z.square(),z.sin()),1);w,_=linear_projection(z,target,np.repeat('a',100))
        self.assertGreater(float((z@w-target).square().mean()),.1)


if __name__=='__main__':main()
