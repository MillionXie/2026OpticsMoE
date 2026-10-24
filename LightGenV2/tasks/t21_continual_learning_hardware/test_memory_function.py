import copy,unittest
import numpy as np
import torch
from .continue_memory_function import partner_bank,mixture,validate_config


class MemoryFunctionTests(unittest.TestCase):
    def test_bank_only_same300_and_pair_offsets(self):
        for task in ('eurosat','clevr','speech_binary'):
            bank=partner_bank(task);self.assertEqual(bank.shape,(8,300));self.assertTrue(np.array_equal(bank,partner_bank(task)))
            for row in bank:self.assertTrue(np.array_equal(np.sort(row),np.arange(300)))
            if task!='eurosat':self.assertTrue(np.array_equal(bank[:,1::2],bank[:,::2]+1))

    def test_positive_encoded_mixture_and_no_original_hard_label_claim(self):
        x=torch.rand(2,224,224);y=torch.rand_like(x);self.assertTrue(torch.equal(mixture(x,y),.8*x+.2*y))
        self.assertGreaterEqual(float(mixture(x,y).min()),0)
        with self.assertRaises(ValueError):mixture(x,y[:1])

    def test_fixed_equal_protocol_only(self):
        old=dict(stage_epochs=[12,8,6,4],optical_lr=.0003,diagnostic_basis='old')
        cfg=dict(old,stage_epochs=[12,8,8,4],old_memory_function_augmentation=dict(mixture_fraction=.2,bank_size=8,old_batch_alternation=2,teacher_target='original corresponding stage on derived TRAIN amplitudes, KL only',hard_label_CE_on_mixtures=False))
        validate_config(old,cfg)
        with self.assertRaises(ValueError):validate_config(old,dict(cfg,optical_lr=.001))
        bad=copy.deepcopy(cfg);bad['old_memory_function_augmentation']['hard_label_CE_on_mixtures']=True
        with self.assertRaises(ValueError):validate_config(old,bad)


if __name__=='__main__':unittest.main()
