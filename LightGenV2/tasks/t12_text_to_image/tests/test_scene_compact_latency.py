import unittest
from ..scene_compact_latency import known_work


class MatchedTimingScope(unittest.TestCase):
    def test_same_cost_scope_for_electronic_and_optical_model(self):
        baseline=known_work(2000,20,0,.8609226935312833)
        lg=known_work(450,20,6,.8609226935312833)
        self.assertEqual(baseline['known_serial_work_ms'],2020)
        self.assertAlmostEqual(lg['known_serial_work_ms'],475.1655361611877)
        self.assertFalse(lg['closed_physical_DAG']);self.assertFalse(lg['physical_measurement'])
        self.assertIn('not assumed zero',lg['unknown_CPU_scheduler_and_communication_cost'])

    def test_cycle_count_is_per_request_not_times_sampling_steps(self):
        for count in [12,36,48,-1]:
            with self.assertRaises(ValueError):known_work(10,1,count,.86)
        with self.assertRaises(ValueError):known_work(-1,1,6,.86)


if __name__=='__main__':unittest.main()
