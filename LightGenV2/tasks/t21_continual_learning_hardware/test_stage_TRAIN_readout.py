import unittest
import numpy as np
import torch
from .fit_measured_stage_TRAIN import join_TRAIN,augment_contrast


class StageTRAINReadoutTests(unittest.TestCase):
    def test_task_ID_join_disambiguates_same_integer(self):
        tasks=np.array(['eurosat','clevr']);ids=np.array([2,2])
        self.assertEqual(join_TRAIN(tasks,ids,tasks[::-1],ids).tolist(),[1,0])

    def test_duplicate_or_changed_ID_rejected(self):
        with self.assertRaises(ValueError):join_TRAIN(np.array(['x','x']),np.array([2,2]),np.array(['x','x']),np.array([2,3]))

    def test_constant_background_preserved(self):
        x=torch.ones(50,784,dtype=torch.float64)
        self.assertTrue(torch.equal(augment_contrast(x,x,torch.zeros_like(x)),x))


if __name__=='__main__':unittest.main()
