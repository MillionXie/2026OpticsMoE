import unittest,json
from pathlib import Path
from .admit_LG_simulation import validate,assess


class AdmissionTests(unittest.TestCase):
    def setUp(self):self.cfg=json.loads(Path(__file__).with_name('config_LG_CD_admission5pp_20261025.json').read_text())
    def test_published_profile_keeps_other_contracts(self):
        validate(self.cfg)
        for key,value in [('lower_correct_delta',-80),('upper_correct_delta',100),('TEST_gradients',True)]:
            with self.assertRaises(ValueError):validate(dict(self.cfg,**{key:value}))
    def test_actual_C_passes_new_but_D_does_not(self):
        scores=lambda d:{n:dict(n=1000,correct=v) for n,v in d.items()}
        self.assertTrue(assess(scores(dict(eurosat=756,clevr=746,speech_binary=765)),'C',self.cfg)['passed'])
        self.assertFalse(assess(scores(dict(eurosat=696,clevr=710,speech_binary=642,physical_binary_raw=750)),'D',self.cfg)['passed'])
    def test_exact_bounds_and_complete_tasks(self):
        d={n:dict(n=1000,correct=v-50) for n,v in self.cfg['original_same1000']['C'].items()}
        self.assertTrue(assess(d,'C',self.cfg)['passed']);d['clevr']['correct']-=1;self.assertFalse(assess(d,'C',self.cfg)['passed'])
        with self.assertRaises(ValueError):assess({'eurosat':d['eurosat']},'C',self.cfg)
        d={n:dict(n=999,correct=v) for n,v in self.cfg['original_same1000']['C'].items()}
        with self.assertRaises(ValueError):assess(d,'C',self.cfg)


if __name__=='__main__':unittest.main()
