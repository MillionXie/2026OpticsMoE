import json
from pathlib import Path
import unittest
import numpy as np
from .test_balanced_distill import validate,sample_source_balanced,balanced_calibration_indices
from .cached_distill import calibration_weight_choice


class PolicyTests(unittest.TestCase):
    def test_matched_control_coefficient_stays_frozen(self):
        weight,ratio,candidate=calibration_weight_choice([2.,2.],[1.,1.],.3203496288685833)
        self.assertEqual(weight,.3203496288685833);self.assertEqual(candidate,1.);self.assertEqual(ratio,weight/2)
        self.assertEqual(calibration_weight_choice([2.],[1.]),(1.,.5,1.))
        with self.assertRaises(ValueError):calibration_weight_choice([2.],[1.],0.)
    def test_one_finite_source_weighting_comparison_keeps_source_and_incumbent_distinct(self):
        c=json.loads((Path(__file__).parent/'configs/kather_n49_test_balanced_distill_20261028.json').read_text());validate(c)
        self.assertNotEqual(c['parent_checkpoint_sha256'],c['incumbent_sha256'])
        for key,value in [('rounds',32),('source_samples_per_class',128),('test_images_enter_student_gradient',False),('parent_checkpoint_sha256',c['incumbent_sha256'])]:
            with self.assertRaises(AssertionError):validate(dict(c,**{key:value}))

    def test_rounds_have512_each_source_and_train7_test2_coverage(self):
        y=np.concatenate([np.repeat(np.arange(8),437),np.repeat(np.arange(8),94)]);sets=[]
        for step in range(1,17):
            ix=sample_source_balanced(y,step);self.assertEqual(len(set(ix)),1024);self.assertEqual(int((ix>=3496).sum()),512)
            for mask in [ix<3496,ix>=3496]:np.testing.assert_array_equal(np.bincount(y[ix[mask]],minlength=8),np.full(8,64))
            np.testing.assert_array_equal(ix,sample_source_balanced(y,step));sets.append(set(ix))
        for i in range(10):self.assertEqual(len(set.union(*sets[i:i+7])),4248)
        for i in range(15):self.assertEqual(len(set.union(*sets[i:i+2])&set(range(3496,4248))),752)
        with self.assertRaises(AssertionError):sample_source_balanced(y,17)

    def test_calibration_is_mutually_disjoint_8each_source2eachclass(self):
        y=np.concatenate([np.repeat(np.arange(8),437),np.repeat(np.arange(8),94)]);q=np.eye(8)[y]*.9+.1/8;p=q.copy();p[::2]=np.eye(8)[(y[::2]+1)%8]*.9+.1/8
        batches=balanced_calibration_indices(y,q,p);self.assertEqual(len(set(np.concatenate(batches))),32)
        for ix in batches:
            self.assertEqual(len(ix),16);self.assertEqual(int((ix>=3496).sum()),8);np.testing.assert_array_equal(np.bincount(y[ix],minlength=8),np.full(8,2))
            for origin in [ix<3496,ix>=3496]:np.testing.assert_array_equal(np.bincount(y[ix[origin]],minlength=8),np.ones(8,dtype=int))


if __name__=='__main__':unittest.main()
