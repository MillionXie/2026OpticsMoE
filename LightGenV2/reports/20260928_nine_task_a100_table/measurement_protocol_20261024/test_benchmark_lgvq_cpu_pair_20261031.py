"""Pure-CPU bookkeeping tests; no Torch import."""
import unittest
import benchmark_lgvq_cpu_pair_20261031 as bench


class Contract(unittest.TestCase):
    def test_mean_is_per_video_not_summed_work(self):
        groups = [dict(workers=w,compute_wall_ms=100/w,process_cpu_ms=100,rows=[dict(thread_cpu_ms=6.25,resize_ms=1,normalization_ms=2,patch_ms=3.25)]*16) for w in (1,16) for _ in range(5)]
        result = bench.summarize(groups)
        self.assertEqual(result['16']['mean_thread_cpu_per_video_ms'],6.25)
        self.assertEqual(result['16']['group_compute_wall_median_ms'],6.25)
        self.assertEqual(result['1']['group_compute_wall_median_ms'],100)


if __name__ == '__main__':
    unittest.main()
