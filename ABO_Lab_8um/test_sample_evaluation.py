import unittest
from sample_evaluation import metrics, sample_indices


class SampleTests(unittest.TestCase):
    def test_fixed_unique_indices(self):
        a=sample_indices(2400,2000,42)
        self.assertEqual(a,sample_indices(2400,2000,42))
        self.assertEqual(len(set(a)),2000)
        self.assertEqual(a,sorted(a))
        self.assertTrue(all(0<=i<2400 for i in a))

    def test_metrics(self):
        m=metrics([{'true_rank':1},{'true_rank':2},{'true_rank':10},{'true_rank':11}])
        self.assertEqual(m['recall_at_1'],0.25)
        self.assertEqual(m['recall_at_5'],0.5)
        self.assertEqual(m['recall_at_10'],0.75)


if __name__=='__main__':unittest.main()
