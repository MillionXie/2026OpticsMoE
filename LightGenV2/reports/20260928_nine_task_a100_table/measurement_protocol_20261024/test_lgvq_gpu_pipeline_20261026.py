import ast
from pathlib import Path
import unittest
from lgvq_gpu_pipeline_20261026 import plan,prefixes,overlap,steady_report,UNIQUE,TOTAL

class Contract(unittest.TestCase):
    def test_same_2048_order(self):
        p=plan();self.assertEqual(len(p),2048);self.assertEqual(len({x[1] for x in p}),512)
        self.assertTrue(all(p[i][1]==i%512 and p[i][2]==i//512 for i in range(2048)))
    def test_complete_fields(self):
        self.assertEqual(TOTAL%16,0);self.assertEqual(UNIQUE%16,0);self.assertEqual(prefixes(),(256,512,1024,2048))
    def test_overlap_union(self):self.assertEqual(overlap([(0,4),(2,7)],3,6),3)
    def test_stability_fixed_rule(self):
        self.assertEqual(steady_report([10*(i+1) for i in range(128)])['first_qualified_prefix_videos'],384)
        self.assertIsNone(steady_report([10,20])['first_qualified_prefix_videos'])
    def test_syntax(self):ast.parse(Path(__file__).with_name('lgvq_gpu_pipeline_20261026.py').read_text())

if __name__=='__main__':unittest.main()
