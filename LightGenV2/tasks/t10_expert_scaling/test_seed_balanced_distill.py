import copy
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from .seed_balanced_distill import validate,config_for_seed,Auxiliary,score_metrics
from .test_balanced_distill import sample_source_balanced,balanced_calibration_indices


class SeedFollowupTests(unittest.TestCase):
    def config(self):return json.loads((Path(__file__).parent/'configs/kather_n49_seed_balanced_followup_20261028.json').read_text())

    def test_only_accepted_seed17_and_two_own_initializations(self):
        c=self.config();validate(c)
        for key,value in [('seeds',[17,27]),('rounds',32),('fixed_lambda_kd',1.)]:
            with self.assertRaises(AssertionError):validate(dict(c,**{key:value}))
        c['_profile']='profile.json'
        with patch.dict(os.environ,T10_SEED_FOLLOWUP_SOURCE_COMMIT='source'):
            for seed in [27,37]:
                v=config_for_seed(c,seed,Path('/run'),False)
                self.assertEqual(v['seed'],seed);self.assertIn(f's{seed}',v['sources']['student']['run'])
                self.assertEqual(v['sources']['student']['sha256'],v['parent_checkpoint_sha256'])
                self.assertNotEqual(v['parent_checkpoint_sha256'],c['seed17_finalized']['checkpoint_sha256'])

    def test_each_seed_balanced_sampling_and_coverage(self):
        y=np.concatenate([np.repeat(np.arange(8),437),np.repeat(np.arange(8),94)]);samples={}
        for seed in [17,27,37]:
            sets=[]
            for step in range(1,17):
                ix=sample_source_balanced(y,step,seed);self.assertEqual(len(set(ix)),1024)
                for origin in [ix<3496,ix>=3496]:np.testing.assert_array_equal(np.bincount(y[ix[origin]],minlength=8),np.full(8,64))
                sets.append(set(ix))
            for start in range(10):self.assertEqual(len(set.union(*sets[start:start+7])),4248)
            for start in range(15):self.assertEqual(len(set.union(*sets[start:start+2])&set(range(3496,4248))),752)
            samples[seed]=sample_source_balanced(y,1,seed)
        self.assertFalse(np.array_equal(samples[27],samples[37]));self.assertFalse(np.array_equal(samples[17],samples[27]))

    def test_D4_and_calibration_use_actual_seed(self):
        y=np.concatenate([np.repeat(np.arange(8),437),np.repeat(np.arange(8),94)]);q=np.eye(8)[y]*.9+.1/8;p=q.copy();p[::2]=np.eye(8)[(y[::2]+1)%8]*.9+.1/8
        codes=[];batches=[]
        for seed in [27,37]:
            aux=object.__new__(Auxiliary);aux.c={'seed':seed};aux.y=y;aux.calibration_view0=False;aux.start_round(1)
            np.testing.assert_array_equal(aux.codes,np.random.default_rng(seed+1000033).integers(0,8,4248));codes.append(aux.codes)
            ix=balanced_calibration_indices(y,q,p,seed);self.assertEqual(len(set(np.concatenate(ix))),32)
            for row in ix:self.assertEqual(int((row>=3496).sum()),8)
            batches.append(ix[0]);aux.calibration_view0=True;aux.start_round(0);self.assertTrue((aux.codes==0).all())
        self.assertFalse(np.array_equal(codes[0],codes[1]));self.assertFalse(np.array_equal(batches[0],batches[1]))

    def test_baseline_metrics_from_locked_source_scores(self):
        y=np.repeat(np.arange(8),94);q=np.eye(8)[y]*.9+.1/8;q[0]=np.eye(8)[1]*.9+.1/8
        m=score_metrics(q,y);self.assertEqual(round(m['accuracy']*752),751);self.assertEqual(sum(np.diag(m['confusion_matrix'])),751)
        with self.assertRaises(AssertionError):score_metrics(q*2,y)


if __name__=='__main__':unittest.main()
