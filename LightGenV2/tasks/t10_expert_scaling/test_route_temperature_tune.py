"""CPU checks for a bounded, validation-only temperature selection."""
import copy
import json
from pathlib import Path
import unittest

from .route_temperature_tune import select_pilot, validate_profile


class TemperatureStudyTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((Path(__file__).parent / 'configs/kather_n49_temperature_pilot_20261027.json').read_text())

    def test_fixed_budget_and_gpu_boundary(self):
        validate_profile(self.cfg)
        for key, value in [('gpu', 1), ('extra_epochs', 30), ('temperatures', [.005, .001]),
                           ('microbatch', 4), ('rollout_seeds', [37])]:
            cfg = copy.deepcopy(self.cfg)
            cfg[key] = value
            with self.assertRaises(ValueError):
                validate_profile(cfg)

    def test_test_scores_cannot_choose_recipe(self):
        candidates = [dict(val=dict(accuracy=590/752, macro_nll=.6), test=1.),
                      dict(val=dict(accuracy=595/752, macro_nll=.7), test=0.)]
        best, rollout = select_pilot(candidates, 590/752, self.cfg)
        self.assertIs(best, candidates[1])
        self.assertTrue(rollout)

    def test_four_validation_examples_required_for_rollout(self):
        for improvement, expected in [(3, False), (4, True)]:
            _, rollout = select_pilot([dict(val=dict(accuracy=(590+improvement)/752, macro_nll=.6))],
                                     590/752, self.cfg)
            self.assertEqual(rollout, expected)

    def test_nll_breaks_accuracy_tie(self):
        candidates = [dict(val=dict(accuracy=.8, macro_nll=.7)),
                      dict(val=dict(accuracy=.8, macro_nll=.6))]
        best, _ = select_pilot(candidates, .78, self.cfg)
        self.assertIs(best, candidates[1])


if __name__ == '__main__':
    unittest.main()
