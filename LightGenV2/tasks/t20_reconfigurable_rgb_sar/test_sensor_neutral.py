import unittest,json
from pathlib import Path
import torch
from .sensor_neutral import encode
from .unified_native import encode_single


class SensorNeutralTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)
    def test_RGB_contract_unchanged(self):
        x=torch.randint(1,255,(2,56,56,3),dtype=torch.uint8);f=torch.rand(2,128)
        torch.testing.assert_close(encode(x,f),encode_single(x,torch.zeros(2,dtype=torch.int64),f),rtol=0,atol=0)
    def test_all_channels_preserved_same_layout_without_sensor_argument(self):
        x=torch.zeros(1,56,56,3,dtype=torch.uint8);x[:,:,:,2]=128;f=torch.ones(1,128)
        a=encode(x,f);self.assertGreater(float(a[:,112:,:112].sum()),0.);self.assertEqual(float(a[:,112:,112:].sum()),0.);torch.testing.assert_close(a.square().sum((1,2)),torch.ones(1))
    def test_finite_profile_and_locked_cpu_auditor(self):
        from .train_sensor_neutral import validate
        cfg=json.loads(Path(__file__).with_name('config_sensor_neutral_20261025.json').read_text());validate(cfg)
        with self.assertRaises(ValueError):validate(dict(cfg,epochs=99999))
        audit=Path(__file__).with_name('audit_sensor_neutral_CPU.py').read_text();self.assertNotIn('torch.save(',audit);self.assertIn('no_checkpoint_reselection=True',audit)
    def test_folded_head_fitted_after_optics_not_fed_back(self):
        code=Path(__file__).with_name('train_sensor_neutral.py').read_text();self.assertLess(code.index("for epoch in range(1,cfg['epochs']+1)"),code.index('headopt=torch.optim.LBFGS'))
        self.assertLess(code.index("locked_before_TEST.json"),code.index("test=load_arrays(a.data,'test')"))


if __name__=='__main__':unittest.main()
