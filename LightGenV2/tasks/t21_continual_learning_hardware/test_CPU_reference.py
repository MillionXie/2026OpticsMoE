import unittest
from .audit_saved_CPU_reference import comparison


class CPUReferenceTests(unittest.TestCase):
    def test_near_tie_is_disclosed_not_called_legacy_pass(self):
        cpu=dict(labels=[0,1],predictions=[0,1],probabilities=[[.5001,.4999],[.1,.9]])
        gpu=dict(labels=[0,1],predictions=[1,1],probabilities=[[.4999,.5001],[.1,.9]])
        result=comparison(cpu,gpu)
        self.assertFalse(result['legacy_gate_passed']);self.assertEqual(result['differing_local_indices'],[0])
        self.assertEqual(result['CPU_correct'],2);self.assertEqual(result['GPU_correct'],1)

    def test_original_probability_gate_not_relaxed(self):
        cpu=dict(labels=[0],predictions=[0],probabilities=[[.6,.4]])
        gpu=dict(labels=[0],predictions=[0],probabilities=[[.597,.403]])
        self.assertFalse(comparison(cpu,gpu)['legacy_gate_passed'])
        self.assertTrue(comparison(cpu,cpu)['legacy_gate_passed'])

    def test_wrong_data_fails_closed(self):
        a=dict(labels=[0],predictions=[0],probabilities=[[.6,.4]])
        b=dict(labels=[1],predictions=[0],probabilities=[[.6,.4]])
        with self.assertRaises(ValueError):comparison(a,b)


if __name__=='__main__':unittest.main()
