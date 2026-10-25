import json,unittest,numpy as np,torch
from pathlib import Path
from .distill_capacity_matched_stage import equal_task_weights,convex_objective,validate_profile,TRAIN_queries


class CapacityMatchedTests(unittest.TestCase):
    def config(self):return json.loads(Path(__file__).with_name('config_capacity_matched_distillation_20261025.json').read_text())
    def test_frozen_profile(self):
        c=self.config();validate_profile(c)
        for k,v in [('maximum_steps',10000),('new_inference_layers',1),('TRAIN_augmentations',100)]:
            with self.assertRaises(ValueError):validate_profile(dict(c,**{k:v}))
    def test_equal_tasks_not_equal_samples(self):
        t=np.array(['a']*3+['b']*7);w=equal_task_weights(t,.5);self.assertAlmostEqual(float(w[t=='a'].sum()),.25);self.assertAlmostEqual(float(w[t=='b'].sum()),.25)
    def test_convex_soft_objective_has_finite_gradient(self):
        c=self.config();c['cross_entropy_weight']=0.;x=torch.randn(20,10,dtype=torch.float64,requires_grad=True);target=x.detach().clone();loss=convex_objective(x,torch.arange(20)%10,target,equal_task_weights(np.repeat('a',20)),c)
        self.assertLess(abs(float(loss)),1e-10);loss.backward();self.assertTrue(torch.isfinite(x.grad).all())
    def test_only_TRAIN_sources_weak_queries_mean_one(self):
        x=torch.rand(10,12)+.5;x=x/x.mean(1,keepdim=True);q,t=TRAIN_queries(x,np.repeat('a',10),self.config());self.assertEqual(q.shape,(90,12));torch.testing.assert_close(q.mean(1),torch.ones(90));torch.testing.assert_close(q[:10],x)


if __name__=='__main__':unittest.main()
