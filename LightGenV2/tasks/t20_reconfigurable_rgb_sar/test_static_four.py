import json,unittest
from pathlib import Path
import torch
from .static_four import build_static_four,geometry,CONTRACT
from .train_shared_frontend import validate


class StaticFourTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_physical_four_aperture_and_capacity(self):
        m=build_static_four('moe');g=geometry(m)
        self.assertEqual(len(m.first_phase),4);self.assertEqual(m.active_indices.tolist(),[0,1,2,3])
        self.assertEqual(g['optical_parameters'],2165136)
        self.assertEqual(g['readout_parameters'],7840)
        self.assertAlmostEqual(g['expert_aperture_mm'],224*.017,places=8)
        self.assertLess(abs(g['aperture_mm']-478*.017),.003)
        self.assertEqual((m.active,m.canvas,m.router_side),(1016,1102,128))
        self.assertTrue(all(p.requires_grad for p in m.first_phase))
        d=build_static_four('d2nn');self.assertEqual(geometry(d)['optical_parameters'],2064512)
        torch.testing.assert_close(m.shared_head.weight,d.shared_head.weight,atol=0,rtol=0)

    def test_ccd_replay_and_gradients(self):
        for architecture in ('moe','d2nn'):
            m=build_static_four(architecture);x=torch.rand(1,224,224);x=m.normalize(x)
            with torch.no_grad():
                m.calibrate_gains(x);r=m(x,return_debug=True);replay=m(x,measured=r['ccd'])
                torch.testing.assert_close(r['logits'],replay['logits'],atol=0,rtol=0)
            m.zero_grad();m(x)['logits'].square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad))
            if architecture=='moe':self.assertEqual(r['route_power'].shape,(1,4))

    def test_exact_finite_profile(self):
        cfg=json.loads(Path(__file__).with_name('config_static_four_20261025.json').read_text());validate(cfg)
        self.assertEqual(cfg['contract'],CONTRACT);self.assertFalse(cfg['TEST_gradients']);self.assertFalse(cfg['TEST_teacher'])
        with self.assertRaises(ValueError):validate(dict(cfg,epochs=9999))
        code=Path(__file__).with_name('train_shared_frontend.py').read_text()
        self.assertLess(code.index('locked_before_TEST.json'),code.index("test=load_arrays(a.data,'test')"))
    def test_aligned_continuation_pins_one_frontend_and_same_graph(self):
        cfg=json.loads(Path(__file__).with_name('config_static_four_aligned_20261025.json').read_text());validate(cfg)
        self.assertEqual(cfg['contract'],CONTRACT);self.assertTrue(cfg['continuation'])
        self.assertEqual(cfg['frontend_SHA256'],'b3a5c165cceeb4fb3329d951ddecbe7a03d4804c1dcecb2f081081ff461cf083')
        self.assertEqual(cfg['cases'],['MoE_AB','D2NN_RGB','D2NN_AB_control'])
        self.assertFalse(cfg['TEST_gradients']);self.assertFalse(cfg['TEST_teacher'])
        audit=Path(__file__).with_name('audit_sensor_neutral_CPU.py').read_text()
        self.assertIn('build_static_four if four',audit);self.assertIn('checked_features(data,a.frontend,front_sha)',audit)


if __name__=='__main__':unittest.main()
