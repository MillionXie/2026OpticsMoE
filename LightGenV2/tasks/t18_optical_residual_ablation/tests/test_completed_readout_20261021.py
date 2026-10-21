import ast,json,unittest
from pathlib import Path
TASK=Path(__file__).resolve().parents[1]

class CompletedReadoutTests(unittest.TestCase):
    def test_bounded_original_head(self):
        c=json.loads((TASK/'lab_caltech20_DVP_completed_readout_profile_20261021.json').read_text())
        self.assertEqual(c['steps'],1200);self.assertEqual(c['fit_seconds'],180)
        self.assertEqual(sum(c['original_TRAIN_counts']),3141)
    def test_source_scope(self):
        s=(TASK/'lab_caltech20_dvp_completed_readout_20261021.py').read_text();ast.parse(s)
        self.assertIn("{'linear.weight','linear.bias'}",s)
        self.assertIn('loss=ce+',s);self.assertIn('F.cross_entropy(z,y',s)
        self.assertIn("endpoint['full_physical']",s)
        self.assertIn('loop.validate_state(',s)
        self.assertIn("out=run/cfg['case']/f'L{a.depth}'",s)
        self.assertNotIn('m.Bench(',s);self.assertNotIn('.cuda(',s)

if __name__=='__main__':unittest.main()
