import unittest
import numpy as np
from .subset import select_groups
from .plan import frame_budget


class TestSubset(unittest.TestCase):
    def test_all_semantic_classes(self):
        strata=np.repeat(np.arange(10),[661,339,100,100,100,100,100,100,100,100])
        ids=select_groups(strata,1000)
        self.assertEqual(len(ids),1000)
        self.assertEqual(len(np.unique(ids)),1000)
        self.assertEqual(np.bincount(strata[ids]).tolist(),[100]*10)

    def test_deterministic(self):
        a=np.repeat(np.arange(8),100)
        np.testing.assert_array_equal(select_groups(a,500),select_groups(a,500))

    def test_pair_preserving_budget(self):
        chosen=select_groups(np.repeat(np.arange(10),100),500)
        indices=(chosen[:,None]*2+np.arange(2)).reshape(-1)
        self.assertEqual(len(indices),1000)
        self.assertTrue(np.all(indices[1::2]==indices[::2]+1))

    def test_small_budget_rejected(self):
        with self.assertRaises(ValueError):select_groups(np.arange(10),9)

    def test_frame_budget(self):
        r=frame_budget()
        self.assertEqual(r['evaluation_frames'],50000)
        self.assertEqual(r['total_with_proposed_TRAIN'],91600)


if __name__=='__main__':unittest.main()
