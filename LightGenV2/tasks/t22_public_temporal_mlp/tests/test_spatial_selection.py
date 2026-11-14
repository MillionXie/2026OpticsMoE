import copy
import unittest

from ..tools.refine_spatial_locality import candidate_score


class SpatialSelectionTests(unittest.TestCase):
    def setUp(self):
        self.config = {'minimum_srcc': .790, 'maximum_srcc': .799,
                       'maximum_slot_share_deviation': .10, 'maximum_global_share_deviation': .05}
        routing = {'maximum_slot_share_deviation': .08, 'selection_share': [.25] * 4}
        self.metrics = {'srcc': .795, 'frame_routing': copy.deepcopy(routing),
                        'video_routing': copy.deepcopy(routing)}
        self.spatial = {name: {'foreign_fraction': .04, 'dark_source_fraction': 0., 'guard_fraction': .02}
                        for name in ('vision', 'language')}

    def test_valid_candidate_prefers_lower_spatial_leakage(self):
        first = candidate_score(self.metrics, self.spatial, self.config)
        self.spatial['vision']['foreign_fraction'] = .02
        self.assertGreater(candidate_score(self.metrics, self.spatial, self.config), first)

    def test_candidate_respects_requested_accuracy_interval_and_balance(self):
        self.metrics['srcc'] = .8
        self.assertIsNone(candidate_score(self.metrics, self.spatial, self.config))
        self.metrics['srcc'] = .795
        self.metrics['frame_routing']['selection_share'] = [.4, .2, .2, .2]
        self.assertIsNone(candidate_score(self.metrics, self.spatial, self.config))

    def test_concentration_score_penalizes_energy_lost_outside_detector_regions(self):
        self.config['spatial_guard_weight'] = 1.
        first = candidate_score(self.metrics, self.spatial, self.config)
        self.spatial['vision']['guard_fraction'] = .04
        self.assertLess(candidate_score(self.metrics, self.spatial, self.config), first)

    def test_accuracy_selects_candidates_once_both_cross_video_targets_are_met(self):
        self.config['spatial_foreign_threshold'] = .01
        for values in self.spatial.values():
            values['foreign_fraction'] = .005
        first = candidate_score(self.metrics, self.spatial, self.config)
        self.metrics['srcc'] = .798
        self.spatial['vision']['foreign_fraction'] = .009
        self.assertGreater(candidate_score(self.metrics, self.spatial, self.config), first)
        self.spatial['vision']['foreign_fraction'] = .02
        self.assertLess(candidate_score(self.metrics, self.spatial, self.config), first)

    def test_each_branch_requires_its_own_positive_phase_gain_when_requested(self):
        self.config['minimum_phase_gain'] = .0005
        self.metrics['phase_gain'] = {'vision': .02, 'language': .001}
        self.assertIsNotNone(candidate_score(self.metrics, self.spatial, self.config))
        for value in (0., -.001, .0004):
            with self.subTest(language_gain=value):
                self.metrics['phase_gain']['language'] = value
                self.assertIsNone(candidate_score(self.metrics, self.spatial, self.config))

    def test_maximum_video_gate_rejects_outliers_despite_small_mean_leakage(self):
        self.config['maximum_video_foreign_fraction'] = .01
        for values in self.spatial.values():
            values.update(foreign_fraction=.005, maximum_video_foreign_fraction=.009)
        self.assertEqual(candidate_score(self.metrics, self.spatial, self.config), self.metrics['srcc'])
        for value in (.01, .012):
            with self.subTest(maximum_foreign=value):
                self.spatial['vision']['maximum_video_foreign_fraction'] = value
                self.assertIsNone(candidate_score(self.metrics, self.spatial, self.config))

    def test_maximum_video_gate_rejects_a_dark_source(self):
        self.config['maximum_video_foreign_fraction'] = .01
        for values in self.spatial.values():
            values.update(foreign_fraction=0., maximum_video_foreign_fraction=0.)
        self.spatial['language']['dark_source_fraction'] = .01
        self.assertIsNone(candidate_score(self.metrics, self.spatial, self.config))


if __name__ == '__main__':
    unittest.main()
