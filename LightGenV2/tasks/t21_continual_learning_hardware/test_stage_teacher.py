import unittest,json
from pathlib import Path
import numpy as np
import torch
from .fit_stage_balanced_teacher import validate_profile,nested_pair_sources,derivatives,full_development_sources,student_checkpoint_key


class StageTeacherTests(unittest.TestCase):
    def test_weak_derivatives_remain_nonidentity_and_default_is_unchanged(self):
        x=torch.rand(16,12)+.3;x=x/x.mean(1,keepdim=True);tasks=np.array(['eurosat']*4+['clevr']*4+['eurosat']*4+['clevr']*4);ids=np.array([0,1,2,3]*4);y=torch.tensor([0,1,0,1]*4)
        torch.manual_seed(4);default=derivatives(x,y,tasks,ids,np.arange(8),np.arange(8,16))[0]
        torch.manual_seed(4);explicit=derivatives(x,y,tasks,ids,np.arange(8),np.arange(8,16),(.02,.05),.001)[0]
        torch.testing.assert_close(default,explicit)
        weak=derivatives(x,y,tasks,ids,np.arange(8),np.arange(8,16),(.002,.005),.0001)[0]
        self.assertFalse(torch.any(torch.all(weak==x[8:].repeat(10,1),1)));torch.testing.assert_close(weak.mean(1),torch.ones(80))
    def test_profiles_are_exact_and_finite(self):
        for stage in ('B800','B800_Euro200','C500'):
            cfg=json.loads(Path(__file__).with_name('config_stage_'+stage+'_teacher_20261025.json').read_text());validate_profile(cfg)
            for key,value in [('teacher_epochs',10000),('SDK',True),('TRAIN_floor',0.)]:
                with self.assertRaises(ValueError):validate_profile(dict(cfg,**{key:value}))
    def test_nested800_preserves500_and_complete_pairs(self):
        ids=np.arange(1000);y=ids%2;old=list(range(500));a=nested_pair_sources(ids,y,old)
        self.assertEqual(len(a),800);self.assertTrue(set(old)<=set(a));self.assertEqual(a,nested_pair_sources(ids,y,old))
        self.assertTrue(all((v^1) in a for v in a))
        with self.assertRaises(ValueError):nested_pair_sources(ids,y,list(range(499))+[501])
    def test_D_full_source_scope_and_all_four_tasks_are_explicit(self):
        cfg=json.loads(Path(__file__).with_name('config_stage_D2NN_D_allTEST_teacher_20261025.json').read_text());validate_profile(cfg)
        self.assertEqual(cfg['stage'],4);self.assertEqual(sum(cfg['sources_per_task'].values()),4000)
        self.assertFalse(cfg['SDK']);self.assertFalse(cfg['GPU']);self.assertTrue(cfg['all_TEST_sources_authorized'])
        self.assertEqual(len(full_development_sources(np.arange(1000),np.arange(1000)%2,'physical_binary_raw')),1000)
        with self.assertRaises(ValueError):validate_profile(dict(cfg,all_TEST_sources_authorized=False))
        scores={n:dict(n=1000,correct=round(v*1000)) for n,v in cfg['development_floor'].items()}
        self.assertEqual(student_checkpoint_key(scores,cfg),(0,0))
        with self.assertRaises(ValueError):student_checkpoint_key({n:v for n,v in scores.items() if n!='clevr'},cfg)
    def test_derivatives_are_nonidentity_mean_one_and_labels_unchanged(self):
        torch.manual_seed(4);x=torch.rand(16,12)+.3;x=x/x.mean(1,keepdim=True)
        tasks=np.array(['eurosat']*4+['clevr']*4+['eurosat']*4+['clevr']*4);ids=np.array([0,1,2,3]*4);y=torch.tensor([0,1,0,1]*4)
        out,labels,tt,ii=derivatives(x,y,tasks,ids,np.arange(8),np.arange(8,16))
        self.assertEqual(out.shape,(80,12));self.assertTrue(torch.equal(labels,y[8:].repeat(10)))
        self.assertTrue(np.array_equal(tt,np.tile(tasks[8:],10)));self.assertTrue(np.array_equal(ii,np.tile(ids[8:],10)))
        self.assertFalse(torch.any(torch.all(out==x[8:].repeat(10,1),1)));torch.testing.assert_close(out.mean(1),torch.ones(80))


if __name__=='__main__':unittest.main()
