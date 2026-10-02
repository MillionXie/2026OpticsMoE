"""Dependency-free entry contract test; full CUDA smoke is run on training host."""
import ast
import json
from pathlib import Path
import unittest


ENTRY = Path(__file__).resolve().parents[1] / 'train_fixed_fusion.py'
CONFIG = ENTRY.with_name('configs') / 'fixed_fusion_rank64_20261002.json'


class Value(float):
    def detach(self):
        return self


class Parameter:
    requires_grad = True

    def requires_grad_(self, enabled):
        self.requires_grad = enabled


class Core:
    fusion_alpha_min, fusion_alpha_max = .4001, .95

    def __init__(self):
        self.block1_optical_fusion_logit = Parameter()
        self.block2_optical_fusion_logit = Parameter()

    def reset_fusion_logits(self, value):
        self.block1_optical_fusion = Value(value)
        self.block2_optical_fusion = Value(value)


class EntryTest(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(ENTRY.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'freeze_fusion')
        scope = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(ENTRY), 'exec'), scope)
        self.freeze = scope['freeze_fusion']

    def test_both_predeclared_candidates_freeze_all_four_existing_gates(self):
        for candidate in json.loads(CONFIG.read_text())['candidates'].values():
            model = type('Model', (), {'language_core': Core(), 'vision_core': Core()})()
            result = self.freeze(model, candidate['alpha'])
            self.assertEqual(len(result), 4)
            for core in (model.language_core, model.vision_core):
                for block in (1, 2):
                    self.assertFalse(getattr(core, f'block{block}_optical_fusion_logit').requires_grad)

    def test_invalid_alpha_rejected(self):
        model = type('Model', (), {'language_core': Core(), 'vision_core': Core()})()
        with self.assertRaises(ValueError):
            self.freeze(model, 1.)


if __name__ == '__main__':
    unittest.main()
