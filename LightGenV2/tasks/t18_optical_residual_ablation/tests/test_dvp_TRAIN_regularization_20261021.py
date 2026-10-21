"""Static admission on local host without a working Torch installation."""
import ast
import json
from pathlib import Path
import unittest

TASK = Path(__file__).resolve().parents[1]


class RegularizationContract(unittest.TestCase):
    def test_profile_and_source(self):
        source = (TASK / 'lab_caltech20_dvp_TRAIN_regularization_20261021.py').read_text()
        ast.parse(source)
        cfg = json.loads((TASK / 'lab_caltech20_DVP_TRAIN_regularization_profile_20261021.json').read_text())
        self.assertEqual(cfg['stage_gate_correct'], 288)
        self.assertGreaterEqual(288 / 411, .70)
        self.assertLess(287 / 411, .70)
        self.assertFalse(cfg['eval_transforms'])
        self.assertFalse(cfg['TEST_gradient'])
        self.assertEqual(cfg['VALID'], 0)
        self.assertGreater(cfg['CCD_additive_sigma_DN'], 0)
        self.assertGreater(cfg['feature_dropout_p'], 0)
        self.assertIn('if not module.training:', source)
        self.assertIn('paired.fit(', source)
        self.assertNotIn('Bench(', source)
        self.assertNotIn('optimizer.step(', source)  # Reuse one published engine.


if __name__ == '__main__':
    unittest.main()
