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
        self.spatial = {name: {'foreign_fraction': .04, 'dark_source_fraction': 0.}
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


if __name__ == '__main__':
    unittest.main()
