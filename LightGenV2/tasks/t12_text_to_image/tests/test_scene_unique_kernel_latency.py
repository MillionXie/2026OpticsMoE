import unittest
from ..scene_unique_kernel_latency import raw_activity_summary


def event(cat,name,ts,dur,correlation=None,**extra):
    return dict(cat=cat,ph='X',name=name,pid=1,tid=2,ts=ts,dur=dur,args={}if correlation is None else{'correlation':correlation},**extra)


class UniqueActivities(unittest.TestCase):
    def test_async_execution_is_classified_by_launch_not_GPU_time(self):
        rows=[event('user_annotation','T12/FFT_simulation',0,20),event('cuda_runtime','cudaLaunchKernel',10,1,7),
            event('cuda_runtime','cudaLaunchKernel',30,1,8),event('kernel','FFT',100,2000,7),event('kernel','readout',110,3000,8)]
        r=raw_activity_summary(rows)
        self.assertEqual(r['groups_ms']['excluded/FFT_simulation'],2)
        self.assertEqual(r['retained_GPU_ms'],3);self.assertEqual(r['device_activity_ms'],5)
    def test_parent_CPU_records_do_not_repeat_GPU_work_and_unknown_work_is_retained(self):
        kernel=event('kernel','gemm',100,3000,9)
        rows=[event('cpu_op','parent',0,200),event('cpu_op','child',1,100),kernel,dict(kernel),event('gpu_memcpy','HtoD',120,1000,10)]
        r=raw_activity_summary(rows)
        self.assertEqual(r['duplicate_raw_records_removed'],1);self.assertEqual(r['raw_unique_GPU_activities'],2)
        self.assertEqual(r['retained_GPU_ms'],4);self.assertEqual(r['groups_ms']['retained_unattributed_GPU'],4)
    def test_different_correlations_are_distinct_even_with_equal_name_and_duration(self):
        r=raw_activity_summary([event('kernel','gemm',10,1000,1),event('kernel','gemm',10,1000,2)])
        self.assertEqual(r['raw_unique_GPU_activities'],2);self.assertEqual(r['device_activity_ms'],2)
    def test_invalid_duration_is_never_dropped(self):
        with self.assertRaises(ValueError):raw_activity_summary([event('kernel','bad',10,-1,1)])


if __name__=='__main__':unittest.main()
