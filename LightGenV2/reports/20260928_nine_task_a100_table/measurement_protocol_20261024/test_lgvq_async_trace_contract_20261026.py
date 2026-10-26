import copy
import unittest
from lgvq_async_trace_contract_20261026 import OPTICAL_PASS_MS,project

def example():
    nodes=[dict(id='front',kind='frontend',duration_ms=3.,parents=[]),
           dict(id='capture',kind='capture_electronics',duration_ms=2.,parents=['front'])]
    previous='capture'
    for i in range(6):
        name=f'optic{i}';nodes.append(dict(id=name,kind='optical_simulation',duration_ms=10.,parents=[previous],field=0));previous=name
    nodes.append(dict(id='head',kind='electronics',duration_ms=1.,parents=[previous]))
    nodes.append(dict(id='token',kind='tokenizer',duration_ms=4.,parents=[]))
    return dict(status='complete',all_predictions_bitwise_equal=True,fields=1,video_calls=16,
                all_startup_and_capture_retained=True,new_inputs_each_field=True,nodes=nodes,
                measured_software_wall_ms=66.)

class ContractTests(unittest.TestCase):
    def test_keeps_startup_capture_and_overlapping_tokenizer(self):
        r=project(example());self.assertAlmostEqual(r['projected_wall_ms'],6+6*OPTICAL_PASS_MS)
        self.assertFalse(r['physical_hardware_time_measured'])
    def test_missing_real_wait_cannot_be_discarded(self):
        t=example();t['measured_software_wall_ms']+=5
        with self.assertRaisesRegex(ValueError,'actual wall'):project(t)
    def test_six_passes_cannot_be_replaced_by_one(self):
        t=example();t['nodes'][2]['kind']='electronics'
        with self.assertRaisesRegex(ValueError,'six'):project(t)
    def test_missing_or_cyclic_dependency_rejected(self):
        t=example();t['nodes'][0]['parents']=['head']
        with self.assertRaisesRegex(ValueError,'dependency'):project(t)
    def test_cached_results_and_discarded_first_fields_rejected(self):
        for key in ['new_inputs_each_field','all_startup_and_capture_retained']:
            t=copy.deepcopy(example());t[key]=False
            with self.assertRaises(ValueError):project(t)

if __name__=='__main__':unittest.main()
