import json,unittest,numpy as np,torch
from pathlib import Path
from .distill_capacity_matched_stage import equal_task_weights,convex_objective,validate_profile,TRAIN_queries


class CapacityMatchedTests(unittest.TestCase):
    def test_conflicting_gradient_projection_is_finite_and_nonconflicting_is_mean(self):
        from .distill_taskwise_conflict import project_conflicts
        a=torch.tensor([1.,0.],dtype=torch.float64);b=torch.tensor([1.,1.],dtype=torch.float64)
        got,r=project_conflicts([a,b],0);torch.testing.assert_close(got,(a+b)/2);self.assertEqual(r['projected_conflicts'],0)
        got,r=project_conflicts([a,-a],0);self.assertTrue(torch.isfinite(got).all());torch.testing.assert_close(got,torch.zeros_like(a));self.assertEqual(r['projected_conflicts'],2)
    def test_provisional_recovery_does_not_relax_absolute_band(self):
        from .recover_linear_assistant import within_bands
        self.assertFalse(within_bands(dict(eurosat=dict(accuracy=.510),clevr=dict(accuracy=.690))))
        self.assertTrue(within_bands(dict(eurosat=dict(accuracy=.510),clevr=dict(accuracy=.720))))
        self.assertFalse(within_bands(dict(eurosat=dict(accuracy=.550),clevr=dict(accuracy=.720))))
    def config(self):return json.loads(Path(__file__).with_name('config_capacity_matched_distillation_20261025.json').read_text())
    def test_frozen_profile(self):
        c=self.config();validate_profile(c)
        for k,v in [('maximum_steps',10000),('new_inference_layers',1),('TRAIN_augmentations',100)]:
            with self.assertRaises(ValueError):validate_profile(dict(c,**{k:v}))
        hard=json.loads(Path(__file__).with_name('config_capacity_matched_hardpair_20261025.json').read_text());validate_profile(hard)
        self.assertEqual(hard['assistant_paired_ranking_weight'],4);self.assertEqual(hard['assistant_derivative_mass'],.75)
    def test_equal_tasks_not_equal_samples(self):
        t=np.array(['a']*3+['b']*7);w=equal_task_weights(t,.5);self.assertAlmostEqual(float(w[t=='a'].sum()),.25);self.assertAlmostEqual(float(w[t=='b'].sum()),.25)
    def test_convex_soft_objective_has_finite_gradient(self):
        c=self.config();c['cross_entropy_weight']=0.;x=torch.randn(20,10,dtype=torch.float64,requires_grad=True);target=x.detach().clone();loss=convex_objective(x,torch.arange(20)%10,target,equal_task_weights(np.repeat('a',20)),c)
        self.assertLess(abs(float(loss)),1e-10);loss.backward();self.assertTrue(torch.isfinite(x.grad).all())
    def test_only_TRAIN_sources_weak_queries_mean_one(self):
        x=torch.rand(10,12)+.5;x=x/x.mean(1,keepdim=True);q,t=TRAIN_queries(x,np.repeat('a',10),self.config());self.assertEqual(q.shape,(90,12));torch.testing.assert_close(q.mean(1),torch.ones(90));torch.testing.assert_close(q[:10],x)


if __name__=='__main__':unittest.main()
