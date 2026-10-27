import unittest
from .ensemble_teacher_cache import admission,d4


class AdmissionTests(unittest.TestCase):
    def test_gate_requires_useful_multiclass_complement(self):
        import numpy as np
        y=np.repeat(np.arange(8),16);teacher=np.eye(8)[y]*.9+.1/8;parent=np.eye(8)[(y+1)%8]*.9+.1/8
        result=admission(y,teacher,parent)
        self.assertTrue(result['admitted']);self.assertEqual(result['teacher_correct_parent_wrong'],128)
        self.assertFalse(admission(y,teacher,teacher)['admitted'])
        concentrated=parent.copy();concentrated[y>=4]=teacher[y>=4]
        self.assertTrue(admission(y,teacher,concentrated)['admitted'])
        concentrated[y>=3]=teacher[y>=3]
        self.assertFalse(admission(y,teacher,concentrated)['admitted'])


class ViewTests(unittest.TestCase):
    def test_eight_unique_views_without_mutating_source(self):
        import torch
        x=torch.arange(9).reshape(1,1,3,3);before=x.clone();views=[d4(x,k) for k in range(8)]
        self.assertEqual(len({tuple(v.flatten().tolist()) for v in views}),8)
        self.assertTrue(torch.equal(x,before));self.assertTrue(torch.equal(x,views[0]))
        with self.assertRaises(ValueError):d4(x,8)


if __name__=='__main__':unittest.main()
