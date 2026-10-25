import unittest
import numpy as np
import torch
from .fit_augmented_residual_teacher import select_sources,ResidualTeacher,augment_train,task_weights,teacher_epochs


class TeacherTests(unittest.TestCase):
    def test_balanced_sources_and_order_invariant(self):
        ids=np.arange(1000);y=np.repeat(np.arange(10),100);a=select_sources(ids,y,'eurosat');q=np.random.default_rng(2).permutation(1000)
        self.assertEqual(a,select_sources(ids[q],y[q],'eurosat'));self.assertEqual(np.bincount(y[a]).tolist(),[20]*10)
    def test_pairs(self):
        ids=np.arange(1000);y=1-ids%2;a=select_sources(ids,y,'clevr')
        self.assertEqual(len(a),200);self.assertEqual(len(set(v//2 for v in a)),100)
    def test_bad_budget(self):
        with self.assertRaises(ValueError):select_sources(np.arange(1000),np.arange(1000)%10,'eurosat',400)
    def test_parent_exact_at_initialization(self):
        w=torch.randn(10,12);x=torch.randn(4,12);m=ResidualTeacher(w).eval();self.assertTrue(torch.equal(m(x),x@w.T))
    def test_background_preserved(self):
        x=1+.01*torch.randn(40,12);v=augment_train(x,x.mean(0),x.std(0));self.assertTrue(torch.allclose(v.mean(1),torch.ones(40),atol=1e-6))
    def test_balanced_tasks(self):
        t=np.array(['a']*3+['b']*10);w=task_weights(t);self.assertAlmostEqual(float(w[t=='a'].sum()),.5,places=6)
    def test_default_budget_unchanged_and_single_extension(self):
        self.assertEqual(teacher_epochs(None),60)
        self.assertEqual(teacher_epochs(dict(schema='T21_offline_teacher_fit_budget_v1',teacher_epochs=240)),240)
        with self.assertRaises(ValueError):teacher_epochs(dict(schema='T21_offline_teacher_fit_budget_v1',teacher_epochs=1000))
    def test_partial500_keeps_pairs_and_gated_budget(self):
        ids=np.arange(1000);y=ids%2;selected=select_sources(ids,y,'clevr',500)
        self.assertEqual(len(selected),500)
        for i in selected:self.assertIn(i^1,selected)
        self.assertEqual(teacher_epochs(dict(schema='T21_offline_teacher_500_gated_v2',teacher_epochs=240,sources_per_task=500,expose_tasks=['clevr'],teacher_floor={'clevr':.710,'eurosat':.483})),240)


if __name__=='__main__':unittest.main()
