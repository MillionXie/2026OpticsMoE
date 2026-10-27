"""CPU-only checks for original control evidence, not a model test."""
import unittest
from temporal_raw_appendix_20261027 import global_owner,original_interval,CONTROL_NAMES,OPTICAL_TEXT
from lgvq_async_trace_contract_20261026 import OPTICAL_PASS_MS


class OriginalControlTests(unittest.TestCase):
    def test_original_blocking_call_is_not_replaced_by_return_overhead(self):
        n=dict(witness=dict(original_call=[10.,35.]),duration_ms=.001)
        self.assertEqual(original_interval(n,{}),[10.,35.])
        self.assertNotEqual(35.-10.,n['duration_ms'])

    def test_original_gap_uses_adjacent_call_observations(self):
        nodes={'left':dict(witness=dict(original_call=[1.,4.])),
               'right':dict(witness=dict(original_call=[6.,8.]))}
        n=dict(witness=dict(boundaries=['left','right']))
        self.assertEqual(original_interval(n,nodes),[4.,6.])

    def test_initial_gap_starts_at_original_zero(self):
        n=dict(witness=dict(boundaries=['origin','right']))
        self.assertEqual(original_interval(n,{'right':dict(witness=dict(original_call=[9.,10.]))}),[0.,9.])

    def test_global_categories_are_distinct_from_part_submissions(self):
        records={0:dict(synchronization_observations=[dict(kind='score_ready')]),
                 4:dict(synchronization_observations=[dict(kind='device_before_buffers')])}
        for node,field,expected in [('f0.token.hostgap',0,'initial_entry'),('f4.token.hostgap',4,'field_entry'),
                ('f4.slotwait',4,'slot_sync'),('f4.finalsync.hostgap',4,'final_sync'),
                ('f0.sync0',0,'score_sync'),('f4.sync0.hostgap',4,'capture_sync'),('f4.capture1.hostgap',4,'capture_gap')]:
            self.assertEqual(global_owner(dict(id=node,field=field),records),expected)
        self.assertIsNone(global_owner(dict(id='f4.submit0',field=4),records))
        self.assertEqual(len(CONTROL_NAMES),7)

    def test_zero_duration_begin_marker_is_not_an_original_action(self):
        self.assertIsNone(global_owner(dict(id='f4.sync0.begin',field=4),{}))

    def test_three_optical_components_keep_fixed_cycle(self):
        self.assertAlmostEqual(1000/1400+.3+.0307,OPTICAL_PASS_MS,places=14)
        self.assertTrue(all(x in OPTICAL_TEXT for x in ['SLM','CCD曝光','CCD读出']))


if __name__=='__main__':unittest.main()
