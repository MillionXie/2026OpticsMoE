"""Budget correction must not silently change the optical/data/loss contract."""
import json
from pathlib import Path
import unittest


class EqualEpochBudgetTest(unittest.TestCase):
    def test_only_budget_and_documentation_differ(self):
        root=Path(__file__).parent
        before=json.loads((root/'config_reference_fullTRAIN_TESTselect_20261025.json').read_text())
        after=json.loads((root/'config_reference_equalepochs_TESTselect_20261025.json').read_text())
        changed={key for key in set(before)|set(after) if before.get(key)!=after.get(key)}
        self.assertEqual(changed,{'schema','patience','max_phase_fit_seconds','teacher_information','diagnostic_basis'})
        self.assertEqual(after['stage_epochs'],[12,4,6,4])
        self.assertGreater(after['patience'],max(after['stage_epochs']))
        self.assertEqual(after['max_phase_fit_seconds'],7200)
        self.assertEqual(after['maximum_GPUs'],2)
        self.assertEqual(after['per_process_GPU_memory_fraction'],.4)
        self.assertFalse(after['TEST_gradients'])
        self.assertFalse(after['SDK'])
        self.assertFalse(after['automatic_capture'])


if __name__=='__main__':unittest.main()
