"""Static protocol checks; real Torch coordinate/gradient test runs on lab host."""
import ast,json,unittest
from pathlib import Path

HERE=Path(__file__).resolve().parents[1]

class Protocol(unittest.TestCase):
    def test_integer_gate(self):
        c=json.loads((HERE/'lab_caltech20_L2_gap15_profile_20261021.json').read_text())
        self.assertEqual(c['target_correct']['2'],331)
        self.assertLessEqual((337-331)/411*100,1.5)
        self.assertGreater((337-330)/411*100,1.5)
        self.assertFalse(c['TEST_gradient']);self.assertEqual(c['VALID'],0)

    def test_finite_conditioned_pair(self):
        c=json.loads((HERE/'lab_caltech20_DVP_conditioned_pair_profile_20261021.json').read_text())
        self.assertEqual(c['targets'],{'4':288,'6':322})
        self.assertLessEqual(c['fit_seconds'],120)
        self.assertLessEqual(c['max_pairs']*c['steps_per_pair'],2560)
        t=ast.parse((HERE/'lab_caltech20_dvp_conditioned_pair_20261021.py').read_text())
        calls=[n.func.attr for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertNotIn('backward',calls)
        self.assertNotIn('capture_stage',calls)
        self.assertIn('original_replay',calls)

if __name__=='__main__':unittest.main()
