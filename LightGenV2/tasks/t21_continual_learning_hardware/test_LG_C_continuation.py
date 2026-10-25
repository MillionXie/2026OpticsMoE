import unittest,json
from pathlib import Path
from .continue_LG_C_optics import validate,native_retention_loss,configure_simulation_trainability,continuation_objective,same_class_replay
from .model import build_model
import torch


class CContinuationTests(unittest.TestCase):
    def test_augmented_replay_keeps_original_classes_and_teacher_frozen(self):
        torch.manual_seed(1);x=torch.rand(12,4,4);labels=torch.tensor([0,1,2]*4);teacher=torch.randn(12,10,requires_grad=True);ix=torch.tensor([0,4,8])
        mx,my,mt=same_class_replay(x,labels,teacher,ix,.1)
        self.assertEqual(mx.shape,(3,4,4));torch.testing.assert_close(my,labels[ix]);self.assertFalse(mt.requires_grad)
        self.assertTrue(bool((mx>=0).all() and (mx<=1).all()))
        cfg=json.loads(Path(__file__).with_name('config_LG_C_augmented_replay_20261025.json').read_text());validate(cfg);self.assertEqual(cfg['memory_records_per_old_task'],300)
    def test_finite_budget_and_original_memory(self):
        c=json.loads(Path(__file__).with_name('config_LG_C_optical_continuation_20261025.json').read_text());validate(c)
        for key,value in [('epochs',20),('memory_records_per_old_task',1000),('TEST_gradients',True)]:
            with self.assertRaises(ValueError):validate(dict(c,**{key:value}))
        native=json.loads(Path(__file__).with_name('config_LG_C_native_B_route_20261025.json').read_text());validate(native)
        self.assertEqual(native['epochs'],1);self.assertEqual(native['memory_records_per_old_task'],300)
        corrected=json.loads(Path(__file__).with_name('config_LG_C_joint_restore_20261025.json').read_text());validate(corrected)
        gated=json.loads(Path(__file__).with_name('config_LG_C_correct_teacher_20261025.json').read_text());validate(gated)

    def test_only_wrong_TRAIN_teacher_KD_is_removed_not_label_CE(self):
        c=json.loads(Path(__file__).with_name('config_LG_C_correct_teacher_20261025.json').read_text())
        student=torch.zeros(1,10,requires_grad=True);teacher=torch.tensor([[10.]+[-10.]*9]);label=torch.tensor([1])
        loss=continuation_objective(student,label,teacher,c);ce=c['reference_CE_weight']*torch.nn.functional.cross_entropy(student,label,label_smoothing=c['label_smoothing'])
        torch.testing.assert_close(loss,ce);loss.backward();self.assertLess(float(student.grad[0,1]),0)

    def test_deployment_freeze_is_explicitly_undone_for_simulation_only(self):
        model=build_model('moe')
        for value in model.parameters():value.requires_grad_(False)
        actual=configure_simulation_trainability(model)
        self.assertTrue({'router_phase','global_phase','shared_head.weight'}<=set(actual))
        self.assertEqual(len(actual),7)
        for i,phase in enumerate(model.first_phase):self.assertEqual(phase.requires_grad,i in model.active_indices[8:12].tolist())

    def test_native_route_preserves_teacher_and_has_gradient(self):
        c=json.loads(Path(__file__).with_name('config_LG_C_native_B_route_20261025.json').read_text())
        teacher=torch.zeros(2,16);teacher[:,:8]=1/8
        logits=torch.zeros(2,10,requires_grad=True);raw=torch.zeros(2,16,requires_grad=True)
        loss=native_retention_loss(dict(logits=logits,route_power=raw.softmax(1)),torch.zeros(2,10),teacher,c);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertGreater(float(raw.grad[:,8:].sum()),0)
        self.assertTrue(torch.equal(teacher[:,:8],torch.full((2,8),1/8)))


if __name__=='__main__':unittest.main()
