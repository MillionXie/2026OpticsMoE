import ast
import json
from pathlib import Path
import unittest

TASK=Path(__file__).resolve().parents[1]


class WhitenedHeadContract(unittest.TestCase):
    def test_targeted_weights_not_TEST_gradient(self):
        c=json.loads((TASK/'lab_caltech20_DVP_whitened_head_targeted_profile_20261021.json').read_text())
        self.assertFalse(c['TEST_gradient']);self.assertEqual(c['TRAIN_error_neighbor_boost'],5)
        self.assertEqual(c['TRAIN_neighbors_per_error'],3)
        self.assertEqual(c['maximum_steps'],160)

    def test_original_head_and_TRAIN_only(self):
        s=(TASK/'lab_caltech20_dvp_whitened_head_20261021.py').read_text();ast.parse(s)
        c=json.loads((TASK/'lab_caltech20_DVP_whitened_head_profile_20261021.json').read_text())
        self.assertFalse(c['TEST_gradient']);self.assertEqual(c['VALID'],0)
        self.assertEqual(c['target_correct'],288);self.assertEqual(c['new_inference_layers'],0)
        self.assertEqual(c['maximum_steps'],160)
        self.assertIn("state['linear.weight'] = torch.from_numpy(ww)",s)
        self.assertIn("transforms = TRAINTransforms(model, '2', sensor_cfg)",s)
        self.assertIn('loop.validate_state(m,model,original',s)
        self.assertNotIn('Bench(',s)


if __name__=='__main__':unittest.main()
