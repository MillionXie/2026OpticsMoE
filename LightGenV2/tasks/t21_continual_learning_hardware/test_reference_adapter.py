import json
from pathlib import Path
import unittest
import numpy as np
import torch

from .train_native import reference_TRAIN_objective, reference_TRAIN_cache, score


class ReferenceAdapterTest(unittest.TestCase):
    def test_teacher_is_frozen_and_student_gets_gradient(self):
        a=torch.randn(8,10,requires_grad=True)
        b=torch.randn(8,10,requires_grad=True)
        config=dict(reference_temperature=2.,label_smoothing=.02,
                    reference_CE_weight=.2,reference_KD_weight=4.)
        reference_TRAIN_objective(a,torch.arange(8),b,config).backward()
        self.assertIsNone(b.grad)
        self.assertGreater(float(a.grad.abs().sum()),0.)

    def test_teacher_cache_uses_only_TRAIN_and_exact_indices(self):
        class Teacher(torch.nn.Module):
            def forward(self,x):return {'logits':x*2}
        train={'eurosat':dict(x=torch.arange(40.).reshape(20,2),manifest={'split':'train'})}
        indices={'eurosat':np.array([1,4,6])}
        result=reference_TRAIN_cache(Teacher(),train,None,indices,['eurosat'],torch.device('cpu'),2)
        self.assertTrue(torch.equal(result['eurosat'],train['eurosat']['x'][[1,4,6]]*2))
        self.assertFalse(result['eurosat'].requires_grad)
        train['eurosat']['manifest']['split']='test'
        with self.assertRaisesRegex(ValueError,'only consume TRAIN'):
            reference_TRAIN_cache(Teacher(),train,None,indices,['eurosat'],torch.device('cpu'),2)

    def test_joint_selection_and_same_budget_profile(self):
        cfg=json.loads(Path(__file__).with_name('config_reference_adapter_TESTselect_20261024.json').read_text())
        self.assertEqual(cfg['selection_split'],'test')
        self.assertFalse(cfg['TEST_gradients'])
        self.assertEqual(cfg['moe_old_task_loss_weight'],1.)
        self.assertEqual(cfg['memory_records_per_old_task'],300)
        self.assertFalse(any(k.startswith(('moe_stage_epochs','d2nn_stage_epochs')) for k in cfg))
        self.assertAlmostEqual(score({'a':{'balanced_accuracy':.8},'b':{'balanced_accuracy':.6}}),.7)


if __name__=='__main__':unittest.main()
