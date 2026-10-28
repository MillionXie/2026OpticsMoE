"""Small fixed-plan/sample adapter checks, no Torch or models."""
import unittest
import sys
import run_baselines_fullscope_20261029 as entry

class Checks(unittest.TestCase):
    def test_missing_only_plan(self):
        plan=entry.missing_plan('lab')
        self.assertEqual(len(plan),21);self.assertEqual(len(set(plan)),21)
        self.assertEqual({x[0] for x in plan},{'qwen','glm','clip','yolo'})
        self.assertFalse(any(m=='qwen' and c==50 for m,c,r in plan))
        for m in ('glm','clip','yolo'):
            for c in (50,100):self.assertEqual([r for mm,cc,r in plan if (mm,cc)==(m,c)],[1,2,3])

    def test_no_rental_or_GPU_import(self):
        self.assertNotIn('torch',sys.modules)
        with self.assertRaises(ValueError):entry.missing_plan('rental')

    def test_fresh_worker_count_and_child_entry(self):
        entry.adapter.ENTRY=entry.ENTRY
        worker=entry.adapter.adapt(entry.base.worker);suite=entry.adapter.adapt(entry.base.suite)
        self.assertIn(512,worker.__code__.co_consts);self.assertNotIn(200,worker.__code__.co_consts)
        self.assertIn(str(entry.ENTRY),suite.__code__.co_filename)

if __name__=='__main__':unittest.main()
