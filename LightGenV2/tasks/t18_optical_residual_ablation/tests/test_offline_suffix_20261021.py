import ast,json,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
class OfflineSuffix(unittest.TestCase):
    def test_error_guided_TRAIN_signature(self):
        s=(HERE/'lab_caltech20_dvp_error_guided_head_20261021.py').read_text();tree=ast.parse(s);fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='focal_correction')
        self.assertNotIn('te',[x.arg for x in fn.args.args]);self.assertIn('importance',[x.arg for x in fn.args.args]);self.assertIn('sample>0',s)
        c=json.loads((HERE/'lab_caltech20_L4_focal_mixup_profile_20261021.json').read_text());self.assertEqual(c['parent_correct'],285);self.assertLessEqual(c['focal']['fit_seconds']+c['mixup']['fit_seconds'],240)
    def test_camera_paused_L6_continuation(self):
        s=(HERE/'lab_caltech20_dvp_L6_offline_continue_20261021.py').read_text();ast.parse(s)
        for forbidden in ('advance.acquire','Camera(','Bench(','engine.run('):self.assertNotIn(forbidden,s)
        self.assertIn('engine.suffix(',s);self.assertIn('not report[\'full_physical\']',s)
        c=json.loads((HERE/'lab_caltech20_L6_offline_extended_profile_20261021.json').read_text());self.assertTrue(c['camera_paused']);self.assertEqual(c['feature_dropout_p'],0);self.assertLessEqual(c['intermediate_fit_seconds'],1800)
    def test_centered_terminal_scope(self):
        c=json.loads((HERE/'lab_caltech20_L4_terminal_centered_profile_20261021.json').read_text());self.assertEqual(c['parent_correct'],285);self.assertLessEqual(c['vector']['fit_seconds'],120)
        s=(HERE/'lab_caltech20_dvp_terminal_reopen_20261021.py').read_text();ast.parse(s)
        self.assertIn('TRAIN_analytic_softmax_gauge',s);self.assertIn('gradient_updates=0',s);self.assertIn('assert error<1e-10',s)
    def test_geometry_only_terminal_scope(self):
        c=json.loads((HERE/'lab_caltech20_L4_geometry_only_profile_20261021.json').read_text());self.assertTrue(c['tail']['freeze_head']);self.assertEqual(c['tail']['feature_dropout_p'],0);self.assertEqual(c['parent_correct'],285)
        s=(HERE/'lab_caltech20_dvp_postCCD_SAM_20261021.py').read_text();self.assertIn("train_keys=allowed-{'linear.weight','linear.bias'}",s);self.assertIn('expected_trainable=10 if freeze_head else 5150',s)
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
    def test_explicit_human_camera_recovery(self):
        c=json.loads((HERE/'lab_caltech20_L6_resume_G2_profile_20261021.json').read_text());self.assertTrue(c['explicit_human_camera_restored']);self.assertEqual(c['dropout_location'],'CCD_pixels')
        s=(HERE/'lab_caltech20_dvp_L6_resume_G2_20261021.py').read_text();ast.parse(s)
        self.assertIn("for stage in ('3','4','5')",s);self.assertIn('advance.acquire(',s);self.assertIn('F.cross_entropy(held[',s);self.assertNotIn('driver_reset',s)
if __name__=='__main__':unittest.main()
