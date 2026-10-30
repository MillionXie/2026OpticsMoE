"""No-model tests of the requested presentation calculations."""
import math,sys,unittest
import collect_lgvq_reorganized_20261030 as c

class Checks(unittest.TestCase):
    def test_parallel_component_union_does_not_double_count_controller(self):
        s=dict(total_s=.004,nodes=[dict(id='gpu',resource='GPU',start_s=0.,end_s=.002),dict(id='optic',resource='optical',start_s=.001,end_s=.004)])
        r=c.energy(s,1)
        self.assertAlmostEqual(r['energy16_j'],41.388*.004+80.358*.003+62.842*.002)
        self.assertAlmostEqual(r['controller_union_ms'],4.)
        self.assertAlmostEqual(r['optical_electronic_overlap_ms'],1.)
        self.assertFalse(r['physical_optical_energy_measured'])
    def test_saved_populations_and_scopes(self):
        d=c.read(c.DEST/'lgvq_reorganized_sources01.json')
        self.assertEqual(len(d['trials']),36)
        self.assertEqual(len({r['report_sha256'] for r in d['trials']}),36)
        self.assertEqual(d['LG_lab']['fields'],96)
        self.assertEqual(d['LG_rental']['fields'],32)
        self.assertFalse(d['rental_numeric_gate_passed'])
        self.assertAlmostEqual(d['LG_lab']['time16_s'],.04594911715143696)
        self.assertAlmostEqual(sum(d['LG_rental']['equivalent_s_per_group'].values()),d['LG_rental']['time16_s'])
        self.assertLess(d['LG_rental']['time16_s'],d['LG_rental_software']['actual_FFT_s_per_16_videos'])
    def test_shared_prompt_not_multiplied_by16(self):
        d=c.read(c.DEST/'lgvq_reorganized_sources01.json')
        for key in ['LG_lab','LG_rental']:
            self.assertEqual(d[key]['native_tokens']/d[key]['fields'],38)
        self.assertEqual(d['common_tokens16'],7136)
        self.assertNotIn('torch',sys.modules)

if __name__=='__main__':unittest.main()
