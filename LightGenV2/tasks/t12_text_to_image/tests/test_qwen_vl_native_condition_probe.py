import unittest
from types import SimpleNamespace

import torch

from ..qwen_vl_native_condition_probe import restore_parameters, train_examples, override_conditions


class NativeConditionProbeTests(unittest.TestCase):
    def test_train_only_and_distinct_target_selection(self):
        rows = []
        for cat in ('lamp', 'pillow'):
            for split, target in [('val', 'x'), ('train', 'a'), ('train', 'a'), ('train', 'b')]:
                rows.append({'category': cat, 'split': split, 'mode': 'object', 'target': {'product_id': target}})
        self.assertEqual(train_examples(rows), [1, 3, 5, 7])

    def test_restore_keeps_trained_precision_and_fails_on_wrong_keys(self):
        model = torch.nn.Linear(2, 1).bfloat16()
        restore_parameters(model, {'weight': torch.ones(1, 2, dtype=torch.float32)})
        self.assertEqual(model.weight.dtype, torch.float32)
        with self.assertRaises(ValueError):
            restore_parameters(model, {'unknown': torch.ones(1)})

    def test_diagnostic_overrides_are_restored_on_failure(self):
        adapter = torch.nn.Linear(2, 2); spatial = torch.nn.Linear(2, 2)
        model = SimpleNamespace(base=SimpleNamespace(adapter=adapter, spatial=spatial))
        x = torch.ones(1, 2); before = adapter(x).detach()
        with self.assertRaises(RuntimeError):
            with override_conditions(model, torch.zeros(1, 2), True):
                self.assertEqual(float(adapter(x).sum()), 0.)
                self.assertEqual(float(spatial(x).detach().sum()), 0.)
                raise RuntimeError('probe')
        self.assertTrue(torch.equal(adapter(x), before))


if __name__ == '__main__':
    unittest.main()
