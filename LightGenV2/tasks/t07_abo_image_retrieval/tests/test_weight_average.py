import copy
import sys
import unittest
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from standalone.weight_average import average_payloads


class WeightAverageTests(unittest.TestCase):
    def payload(self):
        return dict(metadata={'fusion_alpha_min': .4001, 'input_rms': .5},
                    state_dict={'frontend.weight': torch.ones(2),
                                'ids': torch.tensor([1, 2]),
                                'vision.optics.experts.0': torch.tensor([0., 2.]),
                                'readout.weight': torch.tensor([2., 4.])},
                    selection_score=(.99, .99), auxiliary_training_head={'weight': 1})

    def test_average_one_model_no_inherited_score(self):
        a = self.payload(); b = copy.deepcopy(a)
        b['state_dict']['vision.optics.experts.0'] += 2
        b['state_dict']['readout.weight'] += 4
        c = average_payloads(a, b, .25)
        torch.testing.assert_close(c['state_dict']['vision.optics.experts.0'], torch.tensor([.5, 2.5]))
        torch.testing.assert_close(c['state_dict']['readout.weight'], torch.tensor([3., 5.]))
        self.assertNotIn('selection_score', c)
        self.assertNotIn('auxiliary_training_head', c)
        self.assertFalse(c['weight_average']['prediction_ensemble'])
        self.assertTrue(torch.equal(c['state_dict']['frontend.weight'], a['state_dict']['frontend.weight']))
        self.assertEqual(a['state_dict']['readout.weight'].tolist(), [2., 4.])

    def test_reject_incompatible_and_nonfinite(self):
        a = self.payload()
        for mutation in ('metadata', 'frozen', 'shape', 'nan'):
            b = copy.deepcopy(a)
            if mutation == 'metadata': b['metadata']['input_rms'] = .6
            if mutation == 'frozen': b['state_dict']['frontend.weight'][0] = 2
            if mutation == 'shape': b['state_dict']['readout.weight'] = torch.ones(3)
            if mutation == 'nan': b['state_dict']['readout.weight'][0] = float('nan')
            with self.assertRaises(ValueError): average_payloads(a, b)
        for weight in (0, 1, -1, float('nan')):
            with self.assertRaises(ValueError): average_payloads(a, a, weight)


if __name__ == '__main__':
    unittest.main()
