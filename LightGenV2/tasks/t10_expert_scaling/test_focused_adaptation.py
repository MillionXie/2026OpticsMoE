import unittest
import numpy as np
from .focused_adaptation import focused_indices,round_budget

class FocusedSamplingTests(unittest.TestCase):
    def setUp(self):
        self.labels=np.repeat(np.arange(8),437)
        self.scores=np.full((3496,8),.02)
        self.scores[np.arange(3496),self.labels]=.86
        for c in range(8):
            ix=np.arange(c*437,c*437+32)
            self.scores[ix,c]=.1;self.scores[ix,(c+1)%8]=.76

    def test_balanced_unique_hard_replay(self):
        selected,hard=focused_indices(self.scores,self.labels,17,1)
        self.assertEqual(len(np.unique(selected)),1024)
        self.assertEqual(np.bincount(self.labels[selected]).tolist(),[128]*8)
        self.assertEqual(len(hard),512)
        for c in range(8):
            self.assertTrue(set(range(c*437,c*437+32)).issubset(set(hard)))

    def test_uniform_replay_rotates_and_is_reproducible(self):
        a,ha=focused_indices(self.scores,self.labels,17,1)
        b,hb=focused_indices(self.scores,self.labels,17,2)
        a2,_=focused_indices(self.scores,self.labels,17,1)
        self.assertTrue(np.array_equal(a,a2))
        self.assertTrue(np.array_equal(ha,hb))
        self.assertGreater(len(set(a)^set(b)),500)
        self.assertGreater(len(set(a)-set(ha)),0)

    def test_image_budget_and_minimum_window(self):
        for source in range(4,18):
            images,rounds=round_budget(source)
            self.assertLessEqual(rounds*1024,images)
            self.assertGreaterEqual(rounds,10)
            self.assertLessEqual(rounds,20)
        with self.assertRaises(ValueError):round_budget(18)

    def test_nonfinite_and_missing_class_rejected(self):
        scores=self.scores.copy();scores[0,0]=np.nan
        with self.assertRaises(ValueError):focused_indices(scores,self.labels,17,1)
        with self.assertRaises(ValueError):focused_indices(self.scores[:300],self.labels[:300],17,1)

if __name__=='__main__':unittest.main()
