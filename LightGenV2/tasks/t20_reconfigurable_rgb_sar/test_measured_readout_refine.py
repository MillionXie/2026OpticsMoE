import unittest
import numpy as np
import torch
from .refine_measured_readout import transform_from_TRAIN,augment_features,partial_sources,TRAIN_projection


class MeasuredReadoutRefineTests(unittest.TestCase):
    def test_conditioning_folds_into_existing_bias_free_head(self):
        torch.manual_seed(73);x=torch.randn(80,12,dtype=torch.float64)+1
        matrix=transform_from_TRAIN(x,1e-5);u=torch.randn(10,12,dtype=x.dtype)
        torch.testing.assert_close((x@matrix)@u.T,x@(u@matrix.T).T)
        self.assertTrue(torch.isfinite(matrix).all())

    def test_augmentation_uses_centered_variation_not_zeroing_DC(self):
        x=torch.ones(30,12,dtype=torch.float64)
        view=augment_features(x,x.mean(0),x.std(0),dict(centered_dropout=.05,noise_TRAIN_std=.02),73)
        self.assertLess(float((view-x).abs().max()),1e-5)
        self.assertFalse(torch.equal(view,x))

    def test_partial_pool_balanced_bounded_and_deterministic(self):
        domains=np.repeat([0,1],1000);y=torch.arange(10).repeat_interleave(100).repeat(2)
        ids=np.asarray([str(i) for i in range(2000)]);pred=(y+1)%10
        source=partial_sources(ids,y,domains,np.arange(2000),pred,20)
        self.assertEqual(len(source),400);self.assertEqual(len(set(source)),400)
        for d in (0,1):self.assertEqual(np.bincount(y[source[domains[source]==d]],minlength=10).tolist(),[20]*10)
        np.testing.assert_array_equal(source,partial_sources(ids,y,domains,np.arange(2000),pred,20))

    def test_TRAIN_queries_recover_linear_teacher_but_not_erase_lineage(self):
        torch.manual_seed(73);query=torch.randn(80,12,dtype=torch.float64);teacher=torch.randn(10,12,dtype=query.dtype)
        student,receipt=TRAIN_projection(query,teacher)
        self.assertEqual(receipt['rank'],12);torch.testing.assert_close(student,teacher,atol=1e-10,rtol=1e-10)
        with self.assertRaises(RuntimeError):TRAIN_projection(torch.ones(80,12,dtype=query.dtype),teacher)


if __name__=='__main__':unittest.main()
