import unittest
import numpy as np
from .diagnose_measured_expert import survey_indices


class ExpertSurveyTest(unittest.TestCase):
    def test_fixed_cross_class_TRAIN_identities(self):
        labels=np.repeat(np.arange(10),100)
        selected=survey_indices(labels)
        self.assertEqual(selected,[0,50,150,250,350,450,550,650,750,850,950])
        self.assertEqual(set(labels[selected]),set(range(10)))

    def test_reject_incomplete_or_unbalanced_TRAIN(self):
        with self.assertRaises(ValueError):survey_indices(np.zeros(1000,dtype=int))
        with self.assertRaises(ValueError):survey_indices(np.arange(10))


if __name__=='__main__':unittest.main()
