"""Dependency-free scope guards; numerical gradient test runs on lab CPU."""
import ast
import json
from pathlib import Path
import unittest

TASK = Path(__file__).resolve().parents[1]
SOURCE = TASK/'lab_caltech20_dvp_conditioned_head_20261020.py'


class ConditionedHeadContract(unittest.TestCase):
    def test_objective_cannot_take_test_inputs(self):
        tree = ast.parse(SOURCE.read_text())
        objective = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'objective')
        self.assertFalse(any('test' in n.id.lower() for n in ast.walk(objective) if isinstance(n, ast.Name)))
        self.assertEqual([a.arg for a in objective.args.args],
                         ['theta','bias','x','labels','weights','teacher','keep','origin','cfg'])

    def test_only_original_head_is_assigned(self):
        source = SOURCE.read_text()
        self.assertIn("state.update({'linear.weight':w, 'linear.bias':b})", source)
        self.assertIn("if not k.startswith('linear.')", source)
        self.assertIn('core.assess(model, data, fields', source)
        self.assertNotIn('Camera.open', source)
        self.assertNotIn('.cuda(', source)
        self.assertIn("dm['source_cache_SHA256']", source)

    def test_finite_budget_and_separate_targets(self):
        cfg = json.loads((TASK/'lab_caltech20_DVP_conditioned_head_profile_20261020.json').read_text())
        self.assertEqual(sum(cfg['original_TRAIN_counts']), 3141)
        self.assertLessEqual(cfg['maximum_fit_seconds'], 120)
        self.assertLessEqual(cfg['maximum_steps'], 240)
        self.assertEqual(cfg['targets'], {'4':314,'6':322})
        self.assertEqual(cfg['final_targets'], {'4':342,'6':350})

    def test_pair_uses_full_TRAIN_and_existing_direction(self):
        text=SOURCE.read_text()
        self.assertIn('pair_batch(rng,near,pools,hard,len(tr[\'y\']))', text)
        self.assertIn('TRAIN_spectrum_geometry, pair_direction', text)
        self.assertIn('two_original_rows_antisymmetric_other18_fixed=True', text)
        cfg=json.loads((TASK/'lab_caltech20_DVP_pair_head_profile_20261020.json').read_text())
        self.assertEqual(cfg['objective'], 'huber_pair')
        self.assertLessEqual(cfg['max_pairs'],20)
        self.assertLessEqual(cfg['fit_seconds'],120)


if __name__ == '__main__':
    unittest.main()
