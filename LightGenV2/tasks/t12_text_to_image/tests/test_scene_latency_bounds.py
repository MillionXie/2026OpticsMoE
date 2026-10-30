import unittest
from ..scene_latency_bounds import gpu_kernel_union_ms,conservative_components
from ..scene_request_cpu_upper import cpu_record


class ConservativeTiming(unittest.TestCase):
    def test_GPU_union_never_double_counts_overlapping_streams(self):
        events=[dict(cat='kernel',ph='X',ts=a,dur=b)for a,b in [(0,10),(5,10),(20,10)]]
        events.append(dict(cat='cpu_op',ph='X',ts=0,dur=10000))
        self.assertAlmostEqual(gpu_kernel_union_ms(events),.025)
        with self.assertRaises(ValueError):gpu_kernel_union_ms([])

    def test_upper_CPU_can_exceed_wall_when_multiple_threads_work(self):
        r=cpu_record((0,0,0),(.02,.005,.01));self.assertEqual(r['process_CPU_ms'],20.)
        self.assertEqual(r['wall_ms'],10.);self.assertTrue(r['CPU_work_upper_not_CPU_latency'])

    def test_unknown_cost_kept_as_budget_not_silently_zero(self):
        r=conservative_components(2500.,400.,50.)
        self.assertAlmostEqual(r['additional_hardware_overhead_budget_for_4x_ms'],169.8344638388123)
        self.assertFalse(r['physical_measurement']);self.assertIn('unknown',r['extra_hardware_DMA_SDK_cost'])
        self.assertLess(conservative_components(2500,400,100)['conservative_component_ratio_before_extra_hardware'],r['conservative_component_ratio_before_extra_hardware'])


if __name__=='__main__':unittest.main()
