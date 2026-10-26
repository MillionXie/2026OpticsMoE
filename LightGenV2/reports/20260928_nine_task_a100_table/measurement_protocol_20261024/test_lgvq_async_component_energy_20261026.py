import copy
import unittest
from lgvq_async_component_energy_20261026 import component_energy, interval_union
from lgvq_async_observed_dag_20261026 import measured_dag


def schedule(nodes, spans):
    return dict(wall_ms=max(b for a,b in spans), starts_ms={n['id']:s[0] for n,s in zip(nodes,spans)},
                ends_ms={n['id']:s[1] for n,s in zip(nodes,spans)})


class EnergyTests(unittest.TestCase):
    def test_serial_matches_original_approved_formula(self):
        nodes=[dict(id='e',kind='frontend'),dict(id='o',kind='optical_simulation')]
        r=component_energy(nodes,schedule(nodes,[(0,3),(3,9)]),1)
        self.assertAlmostEqual(r['energy_j'],(104.230*3+121.746*6)/1000)

    def test_overlap_counts_controller_and_board_once(self):
        nodes=[dict(id='e1',kind='frontend'),dict(id='e2',kind='electronics'),dict(id='o',kind='optical_simulation')]
        r=component_energy(nodes,schedule(nodes,[(0,3),(1,2),(0,6)]),1)
        self.assertAlmostEqual(r['energy_j'],(41.388*6+80.358*6+62.842*3)/1000)
        self.assertEqual(r['optical_electronic_overlap_ms'],3)

    def test_unclassified_time_or_unremoved_transfer_rejected(self):
        nodes=[dict(id='e',kind='frontend'),dict(id='h',kind='H2D')]
        with self.assertRaises(ValueError):component_energy(nodes,schedule(nodes,[(0,3),(3,4)]),1)
        with self.assertRaises(ValueError):component_energy(nodes,schedule(nodes,[(0,3),(4,4)]),1)

    def test_cannot_hide_parallel_optical_fields(self):
        nodes=[dict(id='a',kind='optical_simulation'),dict(id='b',kind='optical_simulation')]
        with self.assertRaises(ValueError):component_energy(nodes,schedule(nodes,[(0,3),(1,4)]),1)

    def test_interval_unions_never_double_count(self):
        self.assertEqual(interval_union([(0,2),(1,3),(3,4),(5,5)]),[[0,4]])


class ObservedDAGTests(unittest.TestCase):
    def rows(self):
        return [dict(id='dispatch',kind='dispatch',parents=[],observed_start_ms=0.,observed_end_ms=1.,resource='host',witness='host[0:1]'),
                dict(id='front',kind='frontend',parents=['dispatch'],observed_start_ms=1.,observed_end_ms=3.,resource='front',witness='cuda[0]'),
                dict(id='head',kind='electronics',parents=['front'],observed_start_ms=3.,observed_end_ms=4.,resource='body',witness='cuda[1]')]
    def test_every_node_start_reproduced(self):
        result=measured_dag(self.rows(),measured_wall_ms=4.,clock_tolerance_ms=.001)
        self.assertTrue(result['every_original_node_start_reproduced'])
        self.assertFalse(result['arbitrary_wall_padding_added'])
    def test_matching_last_wall_does_not_excuse_missing_front_dependency(self):
        rows=self.rows();rows[1]['parents']=[]
        with self.assertRaisesRegex(ValueError,'dependency'):measured_dag(rows,measured_wall_ms=4.,clock_tolerance_ms=.001)
    def test_no_unwitnessed_padding_or_resource_overlap(self):
        rows=self.rows();rows[-1]['witness']=None
        with self.assertRaisesRegex(ValueError,'evidence'):measured_dag(rows,measured_wall_ms=4.,clock_tolerance_ms=.001)
        rows=self.rows();rows[-1].update(resource='front',observed_start_ms=2.)
        with self.assertRaisesRegex(ValueError,'serial resource'):measured_dag(rows,measured_wall_ms=4.,clock_tolerance_ms=.001)


if __name__=='__main__':unittest.main()
