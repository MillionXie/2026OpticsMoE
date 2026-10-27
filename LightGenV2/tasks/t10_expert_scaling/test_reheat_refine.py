import unittest
import json
from pathlib import Path
from .reheat_refine import rate,degradation_count,validate


class ScheduleTests(unittest.TestCase):
    def test_source_recipe_and_budget_lock(self):
        c=json.loads((Path(__file__).parent/'configs/kather_n49_reheat_20261028.json').read_text())
        validate(c)
        for name,value in [('rounds',3),('seed',27),('peak_lr',.01),('check_rounds',[1,2,3])]:
            with self.assertRaises(AssertionError):validate(dict(c,**{name:value}))

    def test_fixed_reheat_schedule_and_input_budget(self):
        c=dict(peak_lr=.001,end_lr=.00003)
        values=[rate(k,c) for k in range(1,17)]
        self.assertEqual(values[:4],[.0002,.0005,.001,.001])
        self.assertAlmostEqual(values[-1],.00003)
        self.assertTrue(all(a>b for a,b in zip(values[4:],values[5:])))
        self.assertEqual(16*64,1024);self.assertEqual(2*16*1024,32768)
        with self.assertRaises(ValueError):rate(17,c)

    def test_degradation_needs_both_indicators_and_two_fixed_checks(self):
        self.assertEqual(degradation_count(.9,.8,.84,.74,0),1)
        self.assertEqual(degradation_count(.9,.8,.84,.74,1),2)
        self.assertEqual(degradation_count(.9,.8,.84,.79,1),0)
        self.assertEqual(degradation_count(.9,.8,.89,.74,1),0)


if __name__=='__main__':unittest.main()
