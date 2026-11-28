"""Tests for the distinction between Top-2 events, routed power and passive slot transmission."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .report import CORNERS, optical_usage


class OpticalReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'predictions.npz'
        rank = np.array([[0,1],[3,1],[0,1]])
        raw = np.array([[.6,.3,.01,.02],[.01,.31,.04,.55],[0.,0.,0.,0.]])
        selected = raw[np.arange(3)[:,None],rank]
        capture = selected.sum(1)
        power = (selected+1e-12)/(capture[:,None]+2e-12)
        gate = np.zeros((3,4))
        gate[np.arange(3)[:,None],rank] = power
        pre = gate.copy()
        pre[-1] = 0
        self.fields = dict(positions=CORNERS[rank], slot_indices=rank, router_port_energy=raw,
                           dense_power=(raw+1e-12)/(raw.sum(1,keepdims=True)+4e-12), router_capture=capture,
                           power=power, selected_gate_power=gate, pre_fiber_power=pre, transmitted_power=.75*pre)

    def test_dark_image_and_partly_filtered_slots(self):
        np.savez(self.path, **self.fields)
        out = optical_usage(self.path, [0,1,2,3], 3)
        self.assertEqual(out['selection_count'], [2,3,0,1])
        self.assertAlmostEqual(sum(out['inclusion_fraction']), 2.)
        self.assertAlmostEqual(sum(out['fraction_of_two_selections']), 1.)
        self.assertEqual(out['zero_input_images'], 1)
        self.assertAlmostEqual(out['total_transmitted_fraction'], .5)
        self.assertAlmostEqual(out['total_transmittance'], .75)
        self.assertIsNone(out['conditional_power_transmission'][2])

    def test_passive_total_does_not_hide_active_slot_gain(self):
        self.fields['transmitted_power'][0,1] = self.fields['pre_fiber_power'][0,1] + .01
        self.assertLess(self.fields['transmitted_power'][0].sum(), self.fields['pre_fiber_power'][0].sum())
        np.savez(self.path, **self.fields)
        with self.assertRaisesRegex(ValueError, 'unphysical'):
            optical_usage(self.path, [0,1,2,3], 3)

    def test_actual_corners_and_unfiltered_slots(self):
        np.savez(self.path, **self.fields)
        with self.assertRaisesRegex(ValueError, 'unfiltered'):
            optical_usage(self.path, [0,3], 3)
        self.fields['positions'][0,0] = [254,254]
        np.savez(self.path, **self.fields)
        with self.assertRaisesRegex(ValueError, 'corners'):
            optical_usage(self.path, [0,1,2,3], 3)
