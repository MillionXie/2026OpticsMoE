"""CPU contracts for offline accounting; no mutation of live GPU sources."""
import unittest
from LightGenV2.tasks.t06_video_quality_assessment.reports.strict_tf_resource_audit_20261020 import charged_intervals,observed_energy,frame_positions


class Accounting(unittest.TestCase):
    def test_original_and_repeated_evaluation_frame_schemas(self):
        self.assertEqual(frame_positions({'selected_frame_positions':'[10,35,60,86]'}),[10,35,60,86])
        self.assertEqual(frame_positions({'frame_positions':'[[10,35,60,86]]'}),[10,35,60,86])
        with self.assertRaises(AssertionError):frame_positions({})

    def test_frontend_and_terminal_exclusion(self):
        calls=[dict(call_start=0,first_block_start=1,gpu_output=2),
               dict(call_start=4,first_block_start=5,gpu_output=7)]
        intervals,gaps=charged_intervals(calls)
        self.assertEqual(intervals,[(1.,2.),(2.,4.),(5.,7.)])
        self.assertEqual(gaps,[(2.,4.)])
        # A pacing wait [2.1,3.9] is already inside the charged gap.
        self.assertEqual(sum(b-a for a,b in intervals),5.)

    def test_zero_intercall_gap(self):
        intervals,gaps=charged_intervals([dict(call_start=0,first_block_start=1,gpu_output=2),dict(call_start=2,first_block_start=3,gpu_output=4)])
        self.assertEqual(intervals,[(1.,2.),(3.,4.)]);self.assertEqual(gaps,[(2.,2.)])

    def test_board_integration_and_host_once(self):
        raw0=[dict(host_monotonic_s=i/10,watts=100) for i in range(31)]
        raw1=[dict(host_monotonic_s=i/10,watts=200) for i in range(31)]
        intervals=[(.15,.85),(1.2,1.8)]
        seconds=sum(b-a for a,b in intervals)
        self.assertAlmostEqual(observed_energy(intervals,raw0),130)
        self.assertAlmostEqual(sum(observed_energy(intervals,r) for r in (raw0,raw1))+338.2*seconds,829.66)

    def test_stale_telemetry_fails(self):
        with self.assertRaises(AssertionError):observed_energy([(.1,.7)],[dict(host_monotonic_s=0,watts=100)])

    def test_reversed_gap_fails(self):
        with self.assertRaises(AssertionError):charged_intervals([dict(first_block_start=1,gpu_output=3),dict(call_start=2,first_block_start=4,gpu_output=5)])


if __name__=='__main__':unittest.main()
