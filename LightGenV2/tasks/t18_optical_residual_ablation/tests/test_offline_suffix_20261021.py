import ast,json,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
class OfflineSuffix(unittest.TestCase):
    def test_no_camera_start(self):
        s=(HERE/'lab_caltech20_dvp_suffix_stable_20261021.py').read_text();ast.parse(s)
        for forbidden in ('Bench(','Camera(','acquire(','Start('):self.assertNotIn(forbidden,s)
        self.assertIn("model.linear.register_forward_hook",s);self.assertIn('F.cross_entropy(logits,labels',s)
        self.assertIn('full_physical=False',s)
    def test_finite_scope(self):
        c=json.loads((HERE/'lab_caltech20_L6_E2_stable_suffix_profile_20261021.json').read_text())
        self.assertLessEqual(c['epochs'],12);self.assertLessEqual(c['fit_seconds'],900)
        self.assertEqual(sum(c['original_TRAIN_counts']),3141)
        self.assertGreater(c['feature_dropout_p'],0);self.assertLess(c['feature_dropout_p'],.1)
        s=json.loads((HERE/'lab_caltech20_L4_SAM_profile_20261021.json').read_text());self.assertEqual(s['parent_correct'],284)
        self.assertLessEqual(s['SAM']['fit_seconds'],120)
    def test_readout_scope(self):
        s=(HERE/'lab_caltech20_dvp_suffix_readout_20261021.py').read_text();ast.parse(s)
        for forbidden in ('Bench(','Camera(','acquire(','Start('):self.assertNotIn(forbidden,s)
        self.assertIn("allowed={'linear.weight','linear.bias'}",s)
        self.assertIn('full_physical=False',s)
        c=json.loads((HERE/'lab_caltech20_L6_E2_readout_profile_20261021.json').read_text());self.assertLessEqual(c['protected']['fit_seconds'],60)
    def test_L4_priority_original_tail(self):
        s=(HERE/'lab_caltech20_dvp_postCCD_SAM_20261021.py').read_text();ast.parse(s)
        for forbidden in ('Bench(','Camera(','acquire(','Start('):self.assertNotIn(forbidden,s)
        self.assertIn("==5150",s);self.assertIn('torch.equal(p.detach(),v)',s);self.assertIn('torch.cuda.set_rng_state(gpu_rng)',s)
        c=json.loads((HERE/'lab_caltech20_L4_postCCD_SAM_profile_20261021.json').read_text());self.assertEqual(c['parent_correct'],284);self.assertLessEqual(c['tail']['fit_seconds'],600)
        c=json.loads((HERE/'lab_caltech20_L4_postCCD_pixel_SAM_profile_20261021.json').read_text());self.assertEqual(c['tail']['feature_dropout_p'],0);self.assertEqual(c['tail']['CCD_pixel_dropout_p'],.002)
    def test_terminal_coupled_scope(self):
        c=json.loads((HERE/'lab_caltech20_L4_terminal_coupled_profile_20261021.json').read_text());self.assertEqual(c['parent_correct'],284);self.assertLessEqual(c['coupled']['fit_seconds'],120);self.assertEqual(c['coupled']['max_pairs'],20)
if __name__=='__main__':unittest.main()
