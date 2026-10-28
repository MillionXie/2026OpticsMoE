"""No-device tests: units, real clock provenance, strict error rejection."""
import math
from types import SimpleNamespace
import sys
import unittest
import nvml_direct_observer_20261029 as observer

class NVML:
    def nvmlDeviceGetPowerUsage(self,handle):return 65837
    def nvmlDeviceGetUtilizationRates(self,handle):return SimpleNamespace(gpu=37)
    def nvmlDeviceGetMemoryInfo(self,handle):return SimpleNamespace(used=1536*1024**2)

class Checks(unittest.TestCase):
    def test_original_units_and_actual_power_return_clock(self):
        ticks=iter([10.,10.001,10.002]);sample,raw=observer.acquire(NVML(),None,'active',clock=lambda:next(ticks),unix=lambda:100.)
        self.assertEqual(sample.watts,65.837);self.assertEqual(sample.memory_used_mib,1536.)
        self.assertEqual(sample.host_monotonic_s,10.001);self.assertEqual(raw['query_end_monotonic_s'],10.002)

    def test_long_or_invalid_read_rejected(self):
        ticks=iter([10.,10.001,10.4])
        with self.assertRaises(ValueError):observer.acquire(NVML(),None,'active',clock=lambda:next(ticks))
        class Bad(NVML):
            def nvmlDeviceGetUtilizationRates(self,handle):return SimpleNamespace(gpu=7342472)
        with self.assertRaises(ValueError):observer.acquire(Bad(),None,'active')

    def test_no_GPU_import_or_change_to_missing_plan(self):
        self.assertNotIn('torch',sys.modules)
        import run_baselines_direct_nvml_20261029 as entry
        self.assertEqual(len(entry.baseline.missing_plan('lab')),21)
        self.assertNotIn(('qwen',50,1),entry.baseline.missing_plan('lab'))

if __name__=='__main__':unittest.main()
