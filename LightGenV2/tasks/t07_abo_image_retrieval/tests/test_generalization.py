import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization import backward_with_sam, overlay_config
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.io import picture
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.optics import OpticalPath


class GeneralizationTests(unittest.TestCase):
    def test_sam_restores_parameters_and_uses_second_gradient(self):
        p = torch.nn.Parameter(torch.tensor([3., 4.]))
        opt = torch.optim.SGD([p], lr=.1)
        result, audit = backward_with_sam(lambda: {'loss': .5*p.square().sum()}, opt, .2)
        self.assertTrue(torch.equal(p.detach(), torch.tensor([3., 4.])))
        self.assertTrue(torch.allclose(p.grad, torch.tensor([3.12,4.16])))
        opt.step()
        self.assertTrue(torch.allclose(p.detach(), torch.tensor([2.688,3.584])))
        self.assertGreater(audit['loss_gap'], 0)

    def test_sam_exception_restores_exactly(self):
        p = torch.nn.Parameter(torch.tensor([.12345678]))
        before = p.detach().clone()
        opt = torch.optim.SGD([p], lr=.1)
        calls = []
        def closure():
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError('test interrupt')
            return {'loss': p.square().sum()}
        with self.assertRaisesRegex(RuntimeError, 'test interrupt'):
            backward_with_sam(closure, opt, .1)
        self.assertTrue(torch.equal(p.detach(), before))

    def test_sam_replays_noise_and_excludes_frozen_group(self):
        p = torch.nn.Parameter(torch.tensor([1.]))
        frozen = torch.nn.Parameter(torch.tensor([2.]))
        opt = torch.optim.SGD([{'params':[p],'lr':.1},{'params':[frozen],'lr':0.}])
        draws = []
        def closure():
            draws.append(torch.rand(4))
            return {'loss': p.square().sum()+frozen.square().sum()}
        backward_with_sam(closure,opt,.03)
        self.assertTrue(torch.equal(draws[0],draws[1]))
        self.assertEqual(float(frozen),2.)
        self.assertIsNone(frozen.grad)

    def test_zero_rho_one_pass(self):
        p = torch.nn.Parameter(torch.ones(1));calls=[]
        opt=torch.optim.SGD([p],lr=.1)
        def closure():
            calls.append(1)
            return {'loss':p.square().sum()}
        backward_with_sam(closure,opt,0)
        self.assertEqual(len(calls),1)
        self.assertEqual(float(p.grad),2.)

    def test_contain_keeps_top_and_bottom_and_old_mode_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'tall.png'
            a=np.zeros((400,100,3),dtype=np.uint8);a[:60]=[255,0,0];a[-60:]=[0,0,255]
            Image.fromarray(a).save(path)
            preserved=np.array(picture(path,'contain_white'))
            self.assertTrue(np.any(np.all(preserved==[255,0,0],axis=-1)))
            self.assertTrue(np.any(np.all(preserved==[0,0,255],axis=-1)))
            self.assertTrue(np.all(preserved[:,0]==255))
            self.assertTrue(np.array_equal(np.array(picture(path)),np.array(ImageOps.fit(Image.fromarray(a),(224,224),method=Image.Resampling.BICUBIC,centering=(.5,.5)))))

    def test_fullfield_reads_bottom_and_matches_output_shape(self):
        torch.set_num_threads(2)
        optics=OpticalPath()
        intensity=torch.ones(1,478,478,requires_grad=True)
        for mode in ('prefix_rows','fullfield_rows'):
            optics.readout_mode=mode
            result=optics.decode(intensity,77,torch.float32,False)
            self.assertEqual(result.shape,(1,77,192))
        # Pooling footprint is checked directly with nonuniform lower-half signal.
        raw=torch.zeros(1,478,478);raw[:,300:,200:300]=10
        optics.readout_mode='prefix_rows';old=optics.decode(raw,77,torch.float32,False)
        optics.readout_mode='fullfield_rows';new=optics.decode(raw,77,torch.float32,False)
        self.assertFalse(torch.allclose(old,new))

    def test_controls_only_differ_in_sam_and_language_readout(self):
        configs=[overlay_config({'adapt':{},'augmentation':{}},p) for p in ('preserve_adam','preserve_sam','preserve_fullfield_sam')]
        self.assertEqual(configs[0]['adapt'],configs[1]['adapt'])
        self.assertEqual(configs[1]['adapt'],configs[2]['adapt'])
        self.assertEqual(configs[0]['sam_rho'],0)
        self.assertEqual(configs[1]['sam_rho'],configs[2]['sam_rho'])


if __name__ == '__main__':
    unittest.main()
