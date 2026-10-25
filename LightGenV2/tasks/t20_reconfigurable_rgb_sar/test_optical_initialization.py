import json,unittest
from pathlib import Path
import torch
from .static_four import build_static_four,geometry
from .optical_initialization import restart_existing_parameters


class InitializationTests(unittest.TestCase):
    def test_noop_preserves_all_state(self):
        m=build_static_four('moe');old={n:v.clone() for n,v in m.state_dict().items()};self.assertIsNone(restart_existing_parameters(m,{}))
        for n,v in m.state_dict().items():torch.testing.assert_close(v,old[n],atol=0,rtol=0)

    def test_native_codes_deterministic_existing_parameters_and_fixed_gains(self):
        cfg=json.loads(Path(__file__).with_name('config_static_four_optical_restart_20261026.json').read_text());a=build_static_four('moe');b=build_static_four('moe');shape=geometry(a);g=a.display_gains.clone();names={n:p.shape for n,p in a.named_parameters()};r=restart_existing_parameters(a,cfg);restart_existing_parameters(b,cfg)
        self.assertEqual(r['new_parameters'],0);self.assertEqual(geometry(a),shape);self.assertEqual({n:p.shape for n,p in a.named_parameters()},names);torch.testing.assert_close(a.display_gains,g,atol=0,rtol=0)
        for n,p in a.named_parameters():
            torch.testing.assert_close(p,dict(b.named_parameters())[n],atol=0,rtol=0)
            if 'phase' in n:
                codes=(a.phase_fraction(p)*256).detach().round().long();self.assertEqual(int(codes.min()),0);self.assertEqual(int(codes.max()),255)
        self.assertFalse(cfg['recalibrate_gains']);self.assertFalse(cfg['TEST_teacher']);self.assertFalse(cfg['TEST_gradients']);self.assertLessEqual(cfg['epochs'],80)


if __name__=='__main__':unittest.main()
