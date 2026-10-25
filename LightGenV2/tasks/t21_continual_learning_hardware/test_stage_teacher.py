import unittest,json
from pathlib import Path
import numpy as np
import torch
from .fit_stage_balanced_teacher import validate_profile,nested_pair_sources,derivatives


class StageTeacherTests(unittest.TestCase):
    def test_profiles_are_exact_and_finite(self):
        for stage in ('B800','C500'):
            cfg=json.loads(Path(__file__).with_name('config_stage_'+stage+'_teacher_20261025.json').read_text());validate_profile(cfg)
            for key,value in [('teacher_epochs',10000),('SDK',True),('TRAIN_floor',0.)]:
                with self.assertRaises(ValueError):validate_profile(dict(cfg,**{key:value}))
    def test_nested800_preserves500_and_complete_pairs(self):
        ids=np.arange(1000);y=ids%2;old=list(range(500));a=nested_pair_sources(ids,y,old)
        self.assertEqual(len(a),800);self.assertTrue(set(old)<=set(a));self.assertEqual(a,nested_pair_sources(ids,y,old))
        self.assertTrue(all((v^1) in a for v in a))
        with self.assertRaises(ValueError):nested_pair_sources(ids,y,list(range(499))+[501])
    def test_derivatives_are_nonidentity_mean_one_and_labels_unchanged(self):
        torch.manual_seed(4);x=torch.rand(16,12)+.3;x=x/x.mean(1,keepdim=True)
        tasks=np.array(['eurosat']*4+['clevr']*4+['eurosat']*4+['clevr']*4);ids=np.array([0,1,2,3]*4);y=torch.tensor([0,1,0,1]*4)
        out,labels,tt,ii=derivatives(x,y,tasks,ids,np.arange(8),np.arange(8,16))
        self.assertEqual(out.shape,(80,12));self.assertTrue(torch.equal(labels,y[8:].repeat(10)))
        self.assertTrue(np.array_equal(tt,np.tile(tasks[8:],10)));self.assertTrue(np.array_equal(ii,np.tile(ids[8:],10)))
        self.assertFalse(torch.any(torch.all(out==x[8:].repeat(10,1),1)));torch.testing.assert_close(out.mean(1),torch.ones(80))


if __name__=='__main__':unittest.main()
