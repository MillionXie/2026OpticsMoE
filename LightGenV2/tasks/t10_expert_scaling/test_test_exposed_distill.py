import copy
import json
from pathlib import Path
import unittest
import numpy as np
from .test_exposed_distill import validate,scope_config,teacher_gate
from .mixup_curriculum import sample_indices


class PolicyTests(unittest.TestCase):
    def test_exposure_authorization_and_matched_budget(self):
        c=json.loads((Path(__file__).parent/'configs/kather_n49_test_exposed_distill_20261028.json').read_text());validate(c)
        a=scope_config(c,'train_only');b=scope_config(c,'train_test')
        self.assertFalse(a['test_images_enter_student_gradient']);self.assertTrue(b['test_images_enter_student_gradient'])
        self.assertNotEqual(a['calibration_run'],b['calibration_run']);self.assertEqual(a['rounds'],b['rounds'])
        with self.assertRaises(AssertionError):validate(dict(c,explicit_test_training_authorization=False))

    def test_test_fit_teacher_requires_all_three_fixed_floors(self):
        self.assertTrue(teacher_gate(3427,737,715))
        for values in [(3426,752,752),(3496,736,752),(3496,752,714)]:self.assertFalse(teacher_gate(*values))

    def test_sampling_coverage_distinguishes_union_from_original_train(self):
        for n,window in [(437,4),(531,5)]:
            y=np.repeat(np.arange(8),n);sets=[]
            for step in range(1,17):
                ix=sample_indices(y,step,17);self.assertEqual(len(set(ix)),1024)
                np.testing.assert_array_equal(np.bincount(y[ix],minlength=8),np.full(8,128));sets.append(set(ix))
            for i in range(17-window):self.assertEqual(len(set.union(*sets[i:i+window])),8*n)


class SplitTests(unittest.TestCase):
    def test_only_explicit_union_scope_replaces_train_and_keeps_val_disjoint(self):
        import torch
        from unittest.mock import patch
        from .test_exposed_distill import Auxiliary
        y=np.repeat(np.arange(8),531);ids=np.array(['p'+str(i) for i in range(4248)]);x=np.zeros((4248,2,2,3),np.uint8)
        original=(torch.zeros(3496,3,2,2,dtype=torch.uint8),torch.from_numpy(y[:3496]),ids[:3496]);val=(torch.zeros(1,3,2,2),torch.zeros(1,dtype=torch.long),np.array(['val']))
        for exposed in [False,True]:
            a=Auxiliary.__new__(Auxiliary);a.c={'test_images_enter_student_gradient':exposed};n=4248 if exposed else 3496;a.ids=ids[:n];a.y=y[:n]
            splits={'train':original,'val':val}
            with patch('LightGenV2.tasks.t10_expert_scaling.test_exposed_distill.load_pool',return_value=(x,y,ids,None)):a.configure(splits)
            self.assertEqual(len(splits['train'][1]),n);self.assertEqual('original_train' in splits,exposed);self.assertIs(splits['val'],val)


if __name__=='__main__':unittest.main()
