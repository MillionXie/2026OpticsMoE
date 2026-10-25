import json,unittest
from pathlib import Path
from .train_shared_frontend import validate,merge_phase_gradients
import torch


class SharedFrontendTests(unittest.TestCase):
    def test_phase_projection_is_finite_and_descends_both_tasks(self):
        a=torch.tensor([1.,0.]);b=torch.tensor([-1.,1.]);g,changed=merge_phase_gradients(a,b)
        self.assertTrue(changed);self.assertGreaterEqual(float(g@a),0.);self.assertGreaterEqual(float(g@b),0.)
        torch.testing.assert_close(g,torch.tensor([.25,.75]))
        g,changed=merge_phase_gradients(a,a);self.assertFalse(changed);torch.testing.assert_close(g,a)
        g,changed=merge_phase_gradients(a,torch.zeros_like(a));self.assertFalse(changed);self.assertTrue(torch.isfinite(g).all())
    def test_projection_profile_keeps_inference_and_frozen_gain(self):
        cfg=json.loads(Path(__file__).with_name('config_static_four_phase_projection_20261025.json').read_text());validate(cfg)
        self.assertEqual(cfg['cases'],['MoE_AB']);self.assertFalse(cfg['recalibrate_gains']);self.assertLessEqual(cfg['epochs'],20)
        self.assertFalse(cfg['TEST_gradients']);self.assertFalse(cfg['TEST_teacher']);self.assertTrue(cfg['TEST_informed_followup'])
    def test_same_finite_budget_and_no_TEST_training(self):
        cfg=json.loads(Path(__file__).with_name('config_shared_frontend_joint_20261025.json').read_text());validate(cfg)
        with self.assertRaises(ValueError):validate(dict(cfg,epochs=9999))
        self.assertEqual(cfg['cases'],['MoE_AB','D2NN_RGB','D2NN_AB_control']);self.assertFalse(cfg['TEST_gradients']);self.assertFalse(cfg['TEST_teacher'])
    def test_existing_graph_only_and_lock_before_test(self):
        code=Path(__file__).with_name('train_shared_frontend.py').read_text();self.assertNotIn('nn.Linear(',code);self.assertIn("legacy_parity_passed",code);self.assertLess(code.index("locked_before_TEST.json"),code.index("test=load_arrays(a.data,'test')"))


if __name__=='__main__':unittest.main()
