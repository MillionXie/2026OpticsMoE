import unittest
import numpy as np
import torch
from .refine_measured_B_pairwise import pair_indices,objective


class BPairwiseTests(unittest.TestCase):
    def test_pairs_recovered_after_full_capture_permutation(self):
        ids=np.array([103,100,102,101]);y=np.array([0,1,1,0]);p=pair_indices(ids,y)
        self.assertEqual(ids[p].tolist(),[100,101,102,103]);self.assertEqual(y[p].tolist(),[1,0,1,0])
        with self.assertRaises(ValueError):pair_indices(ids[:3],y[:3])

    def test_only_supplied_TRAIN_and_teacher_detached(self):
        x=torch.randn(6,10,dtype=torch.float64,requires_grad=True);teacher=torch.randn_like(x,requires_grad=True)
        y=torch.tensor([4,7,1,0,1,0]);tasks=np.array(['eurosat']*2+['clevr']*4)
        value=objective(x,y,teacher,tasks,torch.tensor([2,3,4,5]));value.backward()
        self.assertTrue(torch.isfinite(x.grad).all());self.assertIsNone(teacher.grad)


if __name__=='__main__':unittest.main()
