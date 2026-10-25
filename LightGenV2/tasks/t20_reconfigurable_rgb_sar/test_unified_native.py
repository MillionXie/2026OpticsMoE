import unittest,json
from pathlib import Path
import numpy as np
import torch
from .unified_native import encode_single,build_unified
from .train_unified_static import validate,balanced_batches,selection
from LightGenV2.tasks.t21_continual_learning_hardware.model import HardwareOptics
from LightGenV2.tasks.t16_zero_phase_ccd_lifelong.frozen_vision import visual_tiles
from LightGenV2.tasks.t16_zero_phase_ccd_lifelong.data import paired_rgb_sar_field


class UnifiedNativeTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(7)
    def test_single_sensor_components_reconstruct_original_T21_pair(self):
        rgb=torch.randint(1,255,(2,56,56,3),dtype=torch.uint8);sar=torch.randint(1,255,(2,56,56,3),dtype=torch.uint8);features=torch.rand(2,128)+.1
        a=encode_single(rgb,torch.zeros(2,dtype=torch.long),features);b=encode_single(sar,torch.ones(2,dtype=torch.long),torch.zeros_like(features))
        self.assertTrue(torch.all(a[:,112:,112:]==0));self.assertTrue(torch.all(b[:,:112]==0));self.assertTrue(torch.all(b[:,:, :112]==0))
        original=paired_rgb_sar_field(rgb.numpy(),sar.numpy());tiles=visual_tiles(rgb.numpy(),features,torch.device('cpu'))
        paired=torch.cat((torch.cat((tiles[:,0],tiles[:,1]),-1),torch.cat((tiles[:,2],original[:,112:,112:]),-1)),-2)
        torch.testing.assert_close((a+b)/2**.5,paired,atol=2e-8,rtol=1e-5)
        torch.testing.assert_close(a.square().sum((1,2)),torch.ones(2));torch.testing.assert_close(b.square().sum((1,2)),torch.ones(2))
    def test_input_contract_has_no_class_label_or_id(self):
        x=torch.randint(1,255,(2,56,56,3),dtype=torch.uint8)
        with self.assertRaises(ValueError):encode_single(x,torch.tensor([0,2]),torch.ones(2,128))
        with self.assertRaises(ValueError):encode_single(x.float(),torch.tensor([0,1]),torch.ones(2,128))
    def test_exact_T21_backend_and_existing_shared_linear(self):
        for arch in ('moe','d2nn'):
            model=build_unified(arch);self.assertIs(type(model),HardwareOptics);self.assertEqual(model.active,1016);self.assertEqual(model.canvas,1056)
            self.assertEqual(model.shared_head.weight.numel(),7840);self.assertIsNone(model.shared_head.bias)
            if arch=='moe':
                self.assertEqual(len(model.first_phase),16);self.assertEqual(int(model.active_count),4);self.assertEqual(sum(p.requires_grad for p in model.first_phase),4)
    @torch.no_grad()
    def test_simulated_CCD_replacement_is_exact(self):
        x=torch.rand(1,224,224);x=x/x.square().sum().sqrt()
        for arch in ('moe','d2nn'):
            model=build_unified(arch).eval();full=model(x,return_debug=True);replay=model(x,measured=full['ccd'])
            torch.testing.assert_close(full['logits'],replay['logits'],atol=0,rtol=0)
    def test_identical_head_initialization_for_both_architectures(self):
        # Same explicit seed must not be affected by optical parameter counts.
        a=build_unified('moe');b=build_unified('d2nn');torch.testing.assert_close(a.shared_head.weight,b.shared_head.weight,atol=0,rtol=0)
    def test_train_domains_and_validation_selection_not_test_targets(self):
        data=dict(domains=np.array([0,1]*8));b=list(balanced_batches(data,[0,1],4,42));self.assertEqual(sorted(np.concatenate(b).tolist()),list(range(16)))
        rgb=list(balanced_batches(data,[0],4,42));self.assertEqual(len(np.concatenate(rgb)),8);self.assertTrue(all(i%2==0 for i in np.concatenate(rgb)))
        r={'scores':{'RGB':dict(accuracy=.8,loss=.4),'SAR':dict(accuracy=.6,loss=.5)}};self.assertAlmostEqual(selection(r,[0,1])[0],.7)
    def test_finite_published_profile(self):
        cfg=json.loads(Path(__file__).with_name('config_unified_static_20261025.json').read_text());validate(cfg)
        for key,value in [('epochs',10000),('TEST_gradients',True),('TEST_teacher',True)]:
            with self.assertRaises(ValueError):validate(dict(cfg,**{key:value}))


if __name__=='__main__':unittest.main()
