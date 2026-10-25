import unittest,numpy as np,torch,json
from pathlib import Path
from .fit_gated_linear_teacher import select_sources,paired_indices,DISCLOSURE,profile_parameters,paired_derivative_noise,full_covariance_transform
from .refine_measured_ce import contrast_transform

class TestGatedTeacher(unittest.TestCase):
    def test_weak_profile_keeps_sources_and_requires_teacher_TRAIN_fit(self):
        cfg=json.loads(Path(__file__).with_name('config_gated_teacher_B500_pairweak_20261025.json').read_text())
        settings=profile_parameters(cfg)
        self.assertEqual(settings['TRAIN_floor'],{'clevr':.9});self.assertEqual(settings['noise'],.001);self.assertEqual(settings['mixes'],(.02,.05))
        for key,value in [('sources_per_task',1000),('teacher_max_LBFGS_steps',1000),('expose_tasks',['clevr','eurosat'])]:
            bad=dict(cfg);bad[key]=value
            with self.assertRaises(ValueError):profile_parameters(bad)
    def test_default_profile_numerics_unchanged(self):
        cfg=json.loads(Path(__file__).with_name('config_gated_teacher_B500_20261025.json').read_text());v=profile_parameters(cfg)
        self.assertEqual(v,dict(steps=160,mixes=(.1,.2),noise=.02,L2=.0001,shared_pair_noise=False,TRAIN_floor={},full_covariance=False))
    def test_full_covariance_coordinates_preserve_function_and_DC(self):
        torch.manual_seed(3);x=(1+.02*torch.randn(120,9,dtype=torch.float64));x=x/x.mean(1,keepdim=True);w=torch.randn(10,9,dtype=torch.float64)
        transform,report=full_covariance_transform(x,contrast_transform(x));z=x@transform.T
        torch.testing.assert_close(z[:,-1],torch.ones(120,dtype=torch.float64),atol=1e-10,rtol=1e-10)
        torch.testing.assert_close(z[:,:-1].T@z[:,:-1]/120,torch.eye(8,dtype=torch.float64),atol=1e-9,rtol=1e-9)
        conditioned=w@torch.linalg.inv(transform)
        torch.testing.assert_close(z@conditioned.T,x@w.T,atol=1e-9,rtol=1e-9)
        self.assertFalse(report['TEST_statistics']);self.assertTrue(report['no_dimension_removed'])
        cfg=json.loads(Path(__file__).with_name('config_gated_teacher_B500_pairwhiten_20261025.json').read_text());self.assertTrue(profile_parameters(cfg)['full_covariance'])
    def test_pair_noise_does_not_add_differential_noise(self):
        torch.manual_seed(5);source=torch.rand(4,9);pairs=[(0,3),(1,2)];noise=paired_derivative_noise(source,torch.ones(9),.001,pairs)
        for a,b in pairs:torch.testing.assert_close(noise[a],noise[b],atol=0,rtol=0)
        self.assertGreater(float(noise.abs().sum()),0)
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
    def test_derived_pairs_follow_ids_not_capture_order(self):
        pairs=paired_indices(np.array([11,8,10,9]),np.array([1,0,0,1]),np.array(['clevr']*4))
        self.assertEqual(pairs,[(1,3),(2,0)])
        with self.assertRaises(ValueError):paired_indices([11,8,10],[1,0,0],['clevr']*3)

if __name__=='__main__':unittest.main()
