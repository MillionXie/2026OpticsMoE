import unittest
from pathlib import Path
import numpy as np
import torch
from .refine_measured_readout import transform_from_TRAIN,augment_features,partial_sources,TRAIN_projection,read,validate_profile,checkpoint_information,reference_relation,requested_relation,relation_matches


class MeasuredReadoutRefineTests(unittest.TestCase):
    def test_predeclared_small_weak_teacher_and_exact_new_relation(self):
        cfg=read(Path(__file__).with_name('config_measured_readout_weak_teacher_20261026.json'));validate_profile(cfg)
        domains=np.repeat([0,1],1000);y=torch.arange(10).repeat_interleave(100).repeat(2)
        ids=np.asarray([str(i) for i in range(2000)]);source=partial_sources(ids,y,domains,np.arange(2000),(y+1)%10,{0:5,1:1})
        self.assertEqual(len(source),60)
        self.assertEqual(np.bincount(y[source[domains[source]==0]],minlength=10).tolist(),[5]*10)
        self.assertEqual(np.bincount(y[source[domains[source]==1]],minlength=10).tolist(),[1]*10)
        self.assertTrue(relation_matches(requested_relation(dict(RGB=dict(correct=778),SAR=dict(correct=571)),'D2NN',cfg)))
        self.assertFalse(relation_matches(requested_relation(dict(RGB=dict(correct=886),SAR=dict(correct=716)),'D2NN',cfg)))
        self.assertFalse(relation_matches(requested_relation(dict(RGB=dict(correct=839),SAR=dict(correct=700)),'LG',cfg)))
        with self.assertRaises(ValueError):validate_profile(dict(cfg,teacher_SAR_sources_per_class=10))

    def test_reduced_profile_changes_only_predeclared_source_budget_and_retention(self):
        base=read(Path(__file__).with_name('config_measured_readout_refine_20261026.json'))
        reduced=read(Path(__file__).with_name('config_measured_readout_TEST10_midPT_20261026.json'))
        validate_profile(base);validate_profile(reduced)
        self.assertEqual(reduced['teacher_sources_per_class_domain'],10)
        for key in base:
            if key!='teacher_sources_per_class_domain':self.assertEqual(reduced[key],base[key])
        with self.assertRaises(ValueError):validate_profile(dict(reduced,steps=9999))

    def test_teacher_gradient_and_student_inheritance_are_distinguished(self):
        self.assertTrue(checkpoint_information(True,True)['TEST_gradients'])
        self.assertFalse(checkpoint_information(True,False)['TEST_gradients'])
        self.assertTrue(checkpoint_information(True,False)['TEST_information_inherited'])
        self.assertFalse(checkpoint_information(False,False)['TEST_information_inherited'])

    def test_requested_relation_is_only_a_report_never_clips_better_scores(self):
        def scores(a,b):return dict(RGB=dict(correct=a),SAR=dict(correct=b))
        self.assertTrue(reference_relation(scores(819,680),'LG')['both_within_2pp_below'])
        self.assertFalse(reference_relation(scores(839,700),'LG')['both_within_2pp_below'])
        result=reference_relation(scores(961,893),'LG')
        self.assertFalse(result['both_within_2pp_below']);self.assertEqual(result['domains']['RGB']['correct'],961)

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
        smaller=partial_sources(ids,y,domains,np.arange(2000),pred,10)
        self.assertEqual(len(smaller),200);self.assertTrue(set(smaller)<=set(source))

    def test_TRAIN_queries_recover_linear_teacher_but_not_erase_lineage(self):
        torch.manual_seed(73);query=torch.randn(80,12,dtype=torch.float64);teacher=torch.randn(10,12,dtype=query.dtype)
        student,receipt=TRAIN_projection(query,teacher)
        self.assertEqual(receipt['rank'],12);torch.testing.assert_close(student,teacher,atol=1e-10,rtol=1e-10)
        with self.assertRaises(RuntimeError):TRAIN_projection(torch.ones(80,12,dtype=query.dtype),teacher)


if __name__=='__main__':unittest.main()
