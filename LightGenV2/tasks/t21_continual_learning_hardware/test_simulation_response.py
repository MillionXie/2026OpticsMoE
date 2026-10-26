import unittest
import numpy as np
import torch
from .distill_simulation_response import aligned_simulation_probabilities,equal_task_soft_KL,stage_contract


class SimulationResponseTests(unittest.TestCase):
    def test_task_weighting_is_fixed_source_mean_not_accuracy_loss(self):
        from .distill_simulation_response import equal_or_weighted_task_mean,cross_task_gradient_cosines
        values=torch.tensor([1.,3.,4.,8.],dtype=torch.float64);tasks=np.array(['a','a','b','b'])
        torch.testing.assert_close(equal_or_weighted_task_mean(values,tasks,['a','b'],{'b':2}),torch.tensor(14/3,dtype=torch.float64))
        w=torch.nn.Parameter(torch.zeros(2,2,dtype=torch.float64));x=torch.tensor([[1.,0.],[1.,0.]],dtype=torch.float64)
        c=cross_task_gradient_cosines(w,x,torch.tensor([0,1]),np.array(['a','b']),['a','b'])
        self.assertLess(c['a']['b'],-.99);self.assertIsNone(w.grad)
    def test_pair_deltas_keep_each_nonidentity_view_and_label_order(self):
        from .distill_simulation_response import derivative_pair_deltas
        x=torch.tensor([[1.,2.],[4.,6.],[10.,20.],[14.,26.]],dtype=torch.float64)
        labels=torch.tensor([0,1,0,1]);ids=np.array([0,1,0,1]);tasks=np.array(['clevr']*4)
        out=derivative_pair_deltas(x,ids,labels,tasks,2,['clevr'])
        torch.testing.assert_close(out['clevr'],torch.tensor([[3.,4.],[4.,6.]],dtype=torch.float64))
        with self.assertRaises(ValueError):derivative_pair_deltas(x,ids,labels,tasks,3,['clevr'])
    def test_gradient_conflict_diagnostic_uses_derivatives_without_step(self):
        import torch
        from .distill_simulation_response import gradient_conflicts
        w=torch.nn.Parameter(torch.zeros(2,2,dtype=torch.float64));before=w.detach().clone();x=torch.eye(2,dtype=torch.float64);tasks=np.array(['a','a'])
        rows=gradient_conflicts(w,x,torch.tensor([0,1]),tasks,x,torch.tensor([1,0]),tasks,['a'])
        self.assertLess(rows['a']['gradient_cosine'],-.99);self.assertTrue(torch.equal(w,before));self.assertIsNone(w.grad)
    def test_exact_C_contract_does_not_inherit_D_relaxation(self):
        self.assertEqual(stage_contract(dict(architecture='moe',stage=3))[2]['clevr'],(696,746))
        self.assertEqual(stage_contract({})[2]['clevr'],(517,617))
        with self.assertRaises(ValueError):stage_contract(dict(architecture='moe',stage=2))
    def test_exact_LG_D_contract_keeps_five_point_goal_and_PT(self):
        import json
        from pathlib import Path
        cfg=json.loads(Path(__file__).with_name('config_LG_D_weak_source_response_20261026.json').read_text())
        arch,stage,bands,warm=stage_contract(cfg);self.assertEqual((arch,stage,warm),('moe',4,'best_checkpoint.pt'))
        self.assertEqual(bands,dict(eurosat=(788,838),clevr=(714,764),speech_binary=(711,761),physical_binary_raw=(725,775)))
        self.assertEqual(cfg['frozen_S1_SHA256'],'030fa9aeb555065f716ed92bfe860e2ec33637ec588f7808840b0fca81b334b1')
        from .fit_stage_balanced_teacher import validate_profile
        teacher=json.loads(Path(__file__).with_name('config_stage_LG_D_allTEST_teacher_20261026.json').read_text());validate_profile(teacher)
        self.assertEqual(sum(teacher['sources_per_task'].values()),4000);self.assertTrue(teacher['all_TEST_sources_authorized'])
        soft=json.loads(Path(__file__).with_name('config_LG_D_soft_response_20261026.json').read_text())
        self.assertEqual(stage_contract(soft),stage_contract(cfg));self.assertFalse(soft['binary_derivative_CE'])
        self.assertEqual(soft['maximum_steps'],120);self.assertEqual(soft['frozen_S1_SHA256'],cfg['frozen_S1_SHA256'])
    def test_identity_alignment_and_unmodified_probabilities(self):
        ids=np.arange(1000)*2;labels=np.arange(1000)%10;p=np.full((1000,10),.01);p[np.arange(1000),labels]=.91;order=np.arange(999,-1,-1)
        score=dict(labels=labels.tolist(),probabilities=p.tolist())
        result=aligned_simulation_probabilities(ids[order],labels[order],ids,score)
        np.testing.assert_array_equal(result,p[order]);np.testing.assert_allclose(p.sum(1),np.ones(1000))
    def test_mismatched_labels_rejected(self):
        p=np.full((1000,10),.1);ids=np.arange(1000)
        with self.assertRaises(ValueError):aligned_simulation_probabilities(ids,np.ones(1000),ids,dict(labels=[0]*1000,probabilities=p))
    def test_duplicate_sources_rejected(self):
        with self.assertRaises(ValueError):aligned_simulation_probabilities(np.zeros(1000),np.zeros(1000),np.arange(1000),dict())
    def test_soft_loss_is_equal_task_and_label_free(self):
        logits=torch.randn(5,10,dtype=torch.float64,requires_grad=True);p=torch.randn(5,10,dtype=torch.float64).softmax(1).detach().requires_grad_();tasks=np.array(['a','a','b','b','b'])
        loss=equal_task_soft_KL(logits,p,tasks,['a','b']);rows=torch.nn.functional.kl_div(logits.log_softmax(1),p,reduction='none').sum(1)
        torch.testing.assert_close(loss,(rows[:2].mean()+rows[2:].mean())/2);loss.backward();self.assertTrue(torch.isfinite(logits.grad).all());self.assertIsNone(p.grad)


if __name__=='__main__':unittest.main()
