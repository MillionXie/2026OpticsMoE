import copy
import json
from pathlib import Path
import unittest

from .extended_adaptation import accumulate_phases, learning_rate, updated_patience, validate


class ExtendedAdaptationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((Path(__file__).parent / 'configs/kather_n49_extended_adaptation_20261027.json').read_text(encoding='utf-8'))

    def test_adaptation_window_prevents_three_epoch_stop(self):
        stale = 0
        first_stop = None
        for epoch in range(1, 18):
            stale = updated_patience(self.cfg, epoch, stale, improved=False)
            if stale >= self.cfg['patience_after_window']:
                first_stop = epoch
                break
        self.assertEqual(first_stop, 16)

    def test_warmup_then_bounded_decay(self):
        values = [learning_rate(self.cfg, step, 17) for step in range(1, 18)]
        self.assertLess(values[0], values[1])
        self.assertLess(values[1], values[2])
        self.assertAlmostEqual(values[2], self.cfg['new_stage_lr'])
        self.assertAlmostEqual(values[-1], self.cfg['new_stage_lr'] * .1)
        self.assertTrue(all(a > b for a, b in zip(values[2:], values[3:])))

    def test_resource_and_budget_constraints(self):
        validate(self.cfg)
        for key, value in [('parallel_jobs', 4), ('max_adaptation_epochs', 200), ('gpus', {'0': 'GPU0'})]:
            cfg = copy.deepcopy(self.cfg)
            cfg[key] = value
            with self.assertRaises(ValueError):
                validate(cfg)

    def test_circular_phase_average_handles_wraparound(self):
        import torch
        sums = {}
        accumulate_phases(sums, [('phase', torch.tensor([179. * torch.pi / 180.]))])
        accumulate_phases(sums, [('phase', torch.tensor([-179. * torch.pi / 180.]))])
        self.assertAlmostEqual(float(sums['phase'].angle().abs()), float(torch.pi), places=5)


if __name__ == '__main__':
    unittest.main()
