"""CPU contracts for offline accounting; no mutation of live GPU sources."""
import unittest
from LightGenV2.tasks.t06_video_quality_assessment.reports.strict_tf_resource_audit_20261020 import charged_intervals,observed_energy,frame_positions,completion_state,verify_historical_anchor,verify_lab_fp32_runtime


class Accounting(unittest.TestCase):
    def test_fp32_accuracy_is_bound_to_the_cost_runtime_not_just_a_folder_name(self):
        cost=dict(checkpoint_sha256='head',assets={'backbone':'sha'},floating_parameter_dtypes=['torch.float32'],sources={'runtime.py':'bound'})
        accuracy=dict(**cost,tf32=False,autocast=False)
        verify_lab_fp32_runtime(cost,accuracy)
        accuracy['floating_parameter_dtypes']=['torch.bfloat16']
        with self.assertRaises(AssertionError):verify_lab_fp32_runtime(cost,accuracy)
        accuracy['floating_parameter_dtypes']=['torch.float32'];accuracy['sources']={'runtime.py':'changed'}
        with self.assertRaises(AssertionError):verify_lab_fp32_runtime(cost,accuracy)

    def test_finalized_qwen_anchor_is_not_replaced_by_a_new_control(self):
        row=dict(baseline='qwen',batch=1,srcc=.7663428036522781,utilization_percent=63.133111480865225,
                 time_ms=1188.3861464262009,energy_j=462.1134010767563,
                 checkpoint_sha256='e102adc6a7b33335c8f49ae1cbeea3302a1a10017d282da8d6091a640276d0d2')
        old=dict(groups=dict(batch1_near_saturation=[row]))
        verify_historical_anchor(old)
        row['utilization_percent']=66.4
        with self.assertRaises(AssertionError):verify_historical_anchor(old)

    def test_complete_costs_do_not_imply_complete_accuracy(self):
        values=[dict(model='qwen') for _ in range(48)]
        self.assertEqual(completion_state(values),'costs48_complete_accuracy_pending')
        for v in values:v['srcc_reference']=0.
        self.assertEqual(completion_state(values),'all48_complete')
        values[0]['target50_reached']=False
        self.assertNotEqual(completion_state(values),'all48_complete')

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
