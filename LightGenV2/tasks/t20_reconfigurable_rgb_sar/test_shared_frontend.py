import json,unittest
from pathlib import Path
from .train_shared_frontend import validate


class SharedFrontendTests(unittest.TestCase):
    def test_same_finite_budget_and_no_TEST_training(self):
        cfg=json.loads(Path(__file__).with_name('config_shared_frontend_joint_20261025.json').read_text());validate(cfg)
        with self.assertRaises(ValueError):validate(dict(cfg,epochs=9999))
        self.assertEqual(cfg['cases'],['MoE_AB','D2NN_RGB','D2NN_AB_control']);self.assertFalse(cfg['TEST_gradients']);self.assertFalse(cfg['TEST_teacher'])
    def test_existing_graph_only_and_lock_before_test(self):
        code=Path(__file__).with_name('train_shared_frontend.py').read_text();self.assertNotIn('nn.Linear(',code);self.assertIn("legacy_parity_passed",code);self.assertLess(code.index("locked_before_TEST.json"),code.index("test=load_arrays(a.data,'test')"))


if __name__=='__main__':unittest.main()
