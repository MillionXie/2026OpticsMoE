import unittest
from types import SimpleNamespace
import torch
from TransferFromElectricity.tasks.t01_object_retrieval.protocol import (
    common_anchor, physical_phase, phase_summary, split_train_validation, stage_at, active_groups)
from TransferFromElectricity.tasks.t01_object_retrieval.models.generator import LoRALinear


class StagedProtocolTest(unittest.TestCase):
    def test_phase_initialization_is_paired_broad_and_not_saturated(self):
        raw = common_anchor(42)
        torch.testing.assert_close(raw,common_anchor(42),rtol=0,atol=0)
        stats = phase_summary(raw,raw)
        self.assertEqual(stats['rms_change_rad'],0.)
        self.assertEqual(stats['sigmoid_saturated_fraction'],0.)
        self.assertGreater(float(physical_phase(raw).std()),1.)
        self.assertTrue(all(.23 < x < .27 for row in stats['dc_power'] for x in row))

    def test_validation_is_stratified_disjoint_and_reproducible(self):
        samples = [SimpleNamespace(sku_index=c,sample_id=f'{c}_{i}') for c in range(3) for i in range(30)]
        train,val = split_train_validation(samples,42,10)
        self.assertEqual(len(train),60)
        self.assertEqual(len(val),30)
        self.assertFalse({s.sample_id for s in train} & {s.sample_id for s in val})
        self.assertEqual([s.sample_id for s in val],[s.sample_id for s in split_train_validation(samples,42,10)[1]])

    def test_stage_boundaries_keep_electronics_frozen_until_joint(self):
        stages = [{'name':'experts','epochs':4},{'name':'optics','epochs':8},{'name':'joint','epochs':8}]
        for epoch,name in ((1,'experts'),(4,'experts'),(5,'optics'),(12,'optics'),(13,'joint'),(20,'joint')):
            self.assertEqual(stage_at(epoch,stages)[0]['name'],name)
        self.assertNotIn('router',active_groups('experts'))
        self.assertIn('router',active_groups('optics'))
        self.assertNotIn('electronic',active_groups('optics'))
        self.assertIn('electronic',active_groups('joint'))

    def test_fp32_lora_updates_frozen_bfloat16_backbone(self):
        torch.manual_seed(42)
        base = torch.nn.Linear(8,8,dtype=torch.bfloat16)
        layer = LoRALinear(base,rank=2)
        layer.lora_a.data = layer.lora_a.data.float()
        layer.lora_b.data = layer.lora_b.data.float()
        inputs = torch.randn(4,8,dtype=torch.bfloat16)
        initial = base(inputs).detach().clone()
        torch.testing.assert_close(layer(inputs),initial,rtol=0,atol=0)
        optimizer = torch.optim.AdamW([layer.lora_a,layer.lora_b],lr=.01)
        layer(inputs).float().square().sum().backward()
        self.assertGreater(float(layer.lora_b.grad.norm()),0.)
        optimizer.step()
        self.assertGreater(float((layer(inputs)-initial).abs().max()),0.)
        self.assertIsNone(base.weight.grad)


if __name__ == '__main__':
    unittest.main()
