import unittest
from pathlib import Path
import numpy as np
import torch
from .refine_measured_TRAIN_cv import PROFILE,validate,group_folds,ridge_delta,cross_validate,relation
from .refine_measured_readout import read


class TrainCVTests(unittest.TestCase):
    def test_exact_finite_profile(self):
        validate(read(Path(__file__).with_name('config_measured_TRAIN_cv_20261026.json')))
        with self.assertRaises(ValueError):validate(dict(PROFILE,steps=9000))

    def test_scene_pairs_stay_in_same_fold(self):
        ids=['AnnualCrop_1:0','AnnualCrop_1:1','SeaLake_2:0','SeaLake_2:1']
        f=group_folds(ids);self.assertEqual(f[0],f[1]);self.assertEqual(f[2],f[3])
        np.testing.assert_array_equal(f,group_folds(ids))

    def test_ridge_correction_reduces_regularized_training_objective(self):
        torch.manual_seed(73);x=torch.randn(30,8,dtype=torch.float64);y=torch.randn(30,10,dtype=x.dtype);prior=torch.randn(10,8,dtype=x.dtype)
        w=ridge_delta(x,y,prior,.1)
        def cost(v):return (x@v.T-y).square().sum()/len(x)+.1*(v-prior).square().sum()
        self.assertLess(float(cost(w)),float(cost(prior)))
        g=(x.T@(x@w.T-y)/len(x)).T+.1*(w-prior)
        self.assertLess(float(g.abs().max()),1e-10)

    def test_cv_deterministic_and_TRAIN_only(self):
        torch.manual_seed(73);x=torch.randn(90,8,dtype=torch.float64);y=torch.arange(90)%10;domains=np.arange(90)%2
        ids=np.asarray([f'scene_{i}:0' for i in range(90)]);prior=torch.zeros(10,8,dtype=x.dtype)
        a,r=cross_validate(x,y,domains,ids,prior,PROFILE)
        b,s=cross_validate(x,y,domains,ids,prior,PROFILE)
        self.assertEqual(a,b);self.assertEqual(r,s);self.assertFalse(r['TEST_used'])
        for row in r['all_candidates']:self.assertEqual(sum(row['counts'].values()),90)

    def test_target_relation_does_not_clamp_higher_results(self):
        def s(a,b):return dict(RGB=dict(correct=a),SAR=dict(correct=b))
        self.assertTrue(relation(s(819,680),'LG')['both_meet'])
        self.assertFalse(relation(s(839,700),'LG')['both_meet'])
        self.assertTrue(relation(s(778,571),'D2NN')['both_meet'])
        self.assertTrue(relation(s(828,621),'D2NN')['both_meet'])
        self.assertFalse(relation(s(886,716),'D2NN')['both_meet'])
        self.assertEqual(relation(s(886,716),'D2NN')['domains']['SAR']['correct'],716)


if __name__=='__main__':unittest.main()
