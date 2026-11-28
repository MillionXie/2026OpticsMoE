"""Tests for the distinction between Top-2 events, routed power and passive slot transmission."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .report import ARMS, CORNERS, PROFILE, SPLITS, optical_usage, read_json, section_summary, verified_gpu


class GPUAssignmentTests(unittest.TestCase):
    def test_actual_config_and_profile_all_match(self):
        profile = read_json(PROFILE)
        expected = profile['gpu_uuids']['37']
        other = profile['gpu_uuids']['53']
        config = {'assigned_gpu_uuid': expected}
        self.assertEqual(verified_gpu({'gpu_uuid': expected}, config, profile, 37), expected)
        for actual, assignment in ((other, expected), (expected, other), (other, other), (None, None)):
            with self.subTest(actual=actual, assignment=assignment), self.assertRaisesRegex(ValueError, 'GPU UUID'):
                verified_gpu({'gpu_uuid': actual}, {'assigned_gpu_uuid': assignment}, profile, 37)

    def test_two_gpu_cohort_checks_within_each_seed(self):
        profile = read_json(PROFILE)
        predictions, runs = {}, []
        scores = {s: {'correct': 1, 'accuracy_percent': 100.} for s in SPLITS}
        arrays = {s: (np.array(['a']), np.array([0]), np.array([True])) for s in SPLITS}
        for seed in (37, 53):
            for arm in ARMS:
                name = f'{seed}_{arm}'
                predictions[name] = arrays
                runs.append(dict(run=name, arm=arm, seed=seed, locked=scores, final=scores, data={},
                                 optimizer={}, optimizer_steps=1, source_commit='test', source_sha256={},
                                 initial_state_sha256=str(seed), minibatch_orders_sha256=str(seed),
                                 gpu_uuid=profile['gpu_uuids'][str(seed)]))
        result = section_summary(runs, predictions, (37,53))
        self.assertEqual(result['matched_gpu_uuid_by_seed'], {int(s): uuid for s,uuid in profile['gpu_uuids'].items()})
        runs[-1]['gpu_uuid'] = profile['gpu_uuids']['37']
        with self.assertRaisesRegex(ValueError, 'Seed 53: paired arms used different GPU UUIDs'):
            section_summary(runs, predictions, (37,53))


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
