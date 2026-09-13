import unittest
from manual_stage import next_stage
from dual_run import STAGES

class ManualStageTests(unittest.TestCase):
    def test_only_next_inputs(self):
        for a,b in zip(STAGES,STAGES[1:]):self.assertEqual(next_stage(a),b)
        self.assertIsNone(next_stage('language_global'))
        with self.assertRaises(ValueError):next_stage('unknown')
