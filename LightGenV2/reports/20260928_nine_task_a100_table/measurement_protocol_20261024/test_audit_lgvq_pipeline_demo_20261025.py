import unittest
from audit_lgvq_pipeline_demo_20261025 import critical_parts, overlap_seconds, union_seconds


class ReviewHelpers(unittest.TestCase):
    def test_union_does_not_sum_workers(self):
        self.assertEqual(union_seconds([(0,4),(1,3),(3,5),(7,8)]),6)

    def test_clip_overlap(self):
        self.assertEqual(overlap_seconds([(0,4),(2,7)],3,6),3)
        self.assertEqual(overlap_seconds([(0,1)],2,3),0)

    def test_exhaustive_nonoverlap_partition(self):
        run=dict(actual_software_E2E_start=0,actual_software_E2E_end=10,actual_software_E2E_seconds=10,
            records=[dict(cpu_ready=3,front_enqueued=4,gpu_done=5,cpu_result_return=6),
                     dict(cpu_ready=7,front_enqueued=8,gpu_done=9,cpu_result_return=10)])
        self.assertEqual(critical_parts(run),dict(wait_input_and_schedule=4,front_host_to_enqueued=2,body_and_sync=2,CPU_result=2))

    def test_reject_reversed_interval(self):
        with self.assertRaises(AssertionError):union_seconds([(2,1)])


if __name__=='__main__':unittest.main()
