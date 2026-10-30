import unittest
from types import SimpleNamespace as N
from LightGenV2.tasks.t12_text_to_image.scene_scoped_latency import billing_group,summarize,PERIOD_MS


class Scope(unittest.TestCase):
    def test_optical_count_not_UNet_steps_or_parallel_requests(self):
        self.assertAlmostEqual(PERIOD_MS,.8609226935312833)
        self.assertAlmostEqual(6*PERIOD_MS,5.1655361611877)
    def test_simulation_and_residual_exclusions_do_not_hide_OEO_or_backend(self):
        self.assertTrue(billing_group('aten::fft',['T12/FFT_simulation']).startswith('excluded'))
        self.assertTrue(billing_group('aten::mm',['T12/electronic_residual']).startswith('excluded'))
        for name in ['readout','UNet','VAE','adapter','softmax','topk','resize','embedding']:
            self.assertEqual(billing_group(name,['T12/request']),'retained_GPU')
        self.assertEqual(billing_group('aten::view',[]),'excluded/pure_layout')
    def test_direct_kernel_links_are_not_inclusive_parent_spans(self):
        parent=N(name='request',cpu_parent=None,device_type='CPU',kernels=[])
        excluded=N(name='T12/FFT_simulation',cpu_parent=parent,device_type='CPU',kernels=[])
        e1=N(name='fft',cpu_parent=excluded,device_type='CPU',kernels=[N(duration=2000,name='fft')])
        e2=N(name='mm',cpu_parent=parent,device_type='CPU',kernels=[N(duration=3000,name='gemm')])
        gpu=N(device_type='CUDA',time_range=N(elapsed_us=lambda:6000))
        q=summarize([parent,excluded,e1,e2,gpu])
        self.assertEqual(q['linked_activity_ms'],5)
        self.assertEqual(q['groups_ms']['excluded/FFT_simulation'],2)
        self.assertEqual(q['retained_GPU_ms'],4) # Unlinked 1ms is paid, not dropped.
        with self.assertRaises(ValueError):summarize([e1,e2,e2,gpu])


if __name__=='__main__':unittest.main()
