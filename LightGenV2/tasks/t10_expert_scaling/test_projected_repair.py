import copy
import unittest
from .projected_repair import acceptable,project_direction


class PolicyTests(unittest.TestCase):
    def test_repair_cannot_buy_complex_harm_or_observer_selection(self):
        baseline={n:{'correct':v,'corrected':0,'new_wrong':0,'nll':1.,'mean_log_margin':0.} for n,v in [('repair_clean',0),('complex_clean',32),('replay_clean',27),('complex_d4',28),('repair_d4',18)]}
        candidate=copy.deepcopy(baseline);candidate['repair_clean'].update(correct=3,corrected=3,nll=.9);candidate['repair_d4']['mean_log_margin']=.01
        self.assertTrue(acceptable(baseline,candidate,baseline))
        for name in ['complex_clean','complex_d4']:
            q=copy.deepcopy(candidate);q[name]['new_wrong']=1;self.assertFalse(acceptable(baseline,q,baseline))
        q=copy.deepcopy(candidate);q['replay_clean']['new_wrong']=2;self.assertFalse(acceptable(baseline,q,baseline))
        q=copy.deepcopy(candidate);q['repair_d4']['mean_log_margin']=-.01;self.assertFalse(acceptable(baseline,q,baseline))


class TorchTests(unittest.TestCase):
    def test_projection_removes_harmful_component_and_normalizes(self):
        import torch
        d=project_direction(torch.tensor([-1.,2.]),[torch.tensor([1.,0.])])
        self.assertEqual(float(d[0]),0.);self.assertGreater(float(d[1]),0.);self.assertAlmostEqual(float(d.square().mean().sqrt()),1.,places=6)
        with self.assertRaises(ValueError):project_direction(torch.tensor([-1.,0.]),[torch.tensor([1.,0.])])


if __name__=='__main__':unittest.main()
