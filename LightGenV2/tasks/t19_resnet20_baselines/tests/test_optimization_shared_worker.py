import copy
import unittest
from LightGenV2.tasks.t19_resnet20_baselines.optimization_shared_worker import validate_shared


class SharedOptimizationTests(unittest.TestCase):
    def state(self):return dict(active={100:3000,200:800},t19_pids=[200],used_mib=4000,total_mib=24000,
        utilization_percent=40,observed_monotonic=20,gpu_uuid='GPU-fixture')
    def test_second_T19_allowed_with_foreign_context_when_stable(self):
        now=self.state();old=copy.deepcopy(now);old['observed_monotonic']=9
        r=validate_shared(now,[200],old)
        self.assertEqual(r['max_T19_per_GPU'],2);self.assertEqual(r['peer_pids'],[200])
        self.assertFalse(r['other_processes_modified'])
    def test_third_T19_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'Two T19'):validate_shared(self.state(),[200,201])
    def test_busy_or_insufficient_memory_rejected(self):
        for key,value in [('utilization_percent',51),('used_mib',19000)]:
            now=self.state();now[key]=value
            with self.assertRaises(RuntimeError):validate_shared(now,[200],self.state())
    def test_foreign_growth_and_short_observation_rejected(self):
        now=self.state();old=copy.deepcopy(now);old['observed_monotonic']=9;old['active'][100]=2800
        with self.assertRaises(RuntimeError):validate_shared(now,[200],old)
        old=copy.deepcopy(now);old['observed_monotonic']=15
        with self.assertRaises(RuntimeError):validate_shared(now,[200],old)
    def test_new_foreign_PID_rejected(self):
        now=self.state();old=copy.deepcopy(now);old['observed_monotonic']=9;old['active'].pop(100)
        with self.assertRaises(RuntimeError):validate_shared(now,[200],old)


if __name__=='__main__':unittest.main()
