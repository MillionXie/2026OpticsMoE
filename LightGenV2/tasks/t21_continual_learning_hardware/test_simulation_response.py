import unittest
import numpy as np
import torch
from .distill_simulation_response import aligned_simulation_probabilities,equal_task_soft_KL,stage_contract


class SimulationResponseTests(unittest.TestCase):
    def test_exact_C_contract_does_not_inherit_D_relaxation(self):
        self.assertEqual(stage_contract(dict(architecture='moe',stage=3))[2]['clevr'],(696,746))
        self.assertEqual(stage_contract({})[2]['clevr'],(517,617))
        with self.assertRaises(ValueError):stage_contract(dict(architecture='moe',stage=4))
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
