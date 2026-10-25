import unittest,numpy as np
from .fit_gated_linear_teacher import select_sources,DISCLOSURE

class TestGatedTeacher(unittest.TestCase):
    def test_balanced_frozen_half_sources(self):
        ids=np.arange(1000);y=np.repeat(np.arange(10),100);chosen=select_sources(ids,y,'eurosat',500)
        self.assertEqual(len(chosen),500);self.assertEqual(np.bincount(y[chosen]).tolist(),[50]*10)
        self.assertEqual(chosen,select_sources(ids[::-1],y[::-1],'eurosat',500))
        with self.assertRaises(ValueError):select_sources(ids,y,'eurosat',1000)
    def test_binary_pairs(self):
        ids=np.arange(1000);y=ids%2;chosen=select_sources(ids,y,'clevr',500)
        self.assertEqual(len(chosen),500)
        for i in chosen:self.assertIn(i^1,chosen)
        with self.assertRaises(ValueError):select_sources(ids,np.zeros(1000,int),'clevr',500)
    def test_disclosure_remains_contaminated(self):
        self.assertFalse(DISCLOSURE['valid_for_formal_test_table']);self.assertTrue(DISCLOSURE['student_inherits_EVAL_information'])
        self.assertEqual(DISCLOSURE['independent_test_samples'],0)

if __name__=='__main__':unittest.main()
