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
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue import dependency_state


class GeneralizationTests(unittest.TestCase):
    def test_electronic_kernel_expansion_preserves_function_and_optics(self):
        from LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization import expand_electronic_context
        from LightGenV2.tasks.t07_abo_image_retrieval.standalone.model import Residual
        state={};nets={}
        for name,vision in [('vision',True),('language',False)]:
            nets[name]=Residual(vision).double().eval()
            for index in (0,1):
                state.update({f'{name}.blocks.{index}.{k}':v.clone() for k,v in nets[name].state_dict().items()})
        state['vision.optics.global_phase']=torch.randn(478,478)
        payload={'metadata':{},'state_dict':state}
        expanded=expand_electronic_context(payload,{'vision':7,'language':7})
        self.assertNotIn('electronic_context_kernels',payload['metadata'])
        for name,v in state.items():
            if 'token_depthwise.weight' not in name:
                self.assertIs(v,expanded['state_dict'][name])
        for name,vision in [('vision',True),('language',False)]:
            net=Residual(vision,7).double().eval()
            prefix=f'{name}.blocks.0.'
            net.load_state_dict({k[len(prefix):]:v for k,v in expanded['state_dict'].items() if k.startswith(prefix)})
            x=torch.randn(2,196 if vision else 77,192,dtype=torch.float64)
            torch.testing.assert_close(net(x),nets[name](x),rtol=1e-12,atol=1e-12)
            net(x).square().mean().backward()
            grad=net.token_depthwise.weight.grad
            self.assertGreater(float(grad[...,0].abs().sum()),0.)
        # Reapplying a metadata contract must not transform already trained weights.
        same=expand_electronic_context(expanded,{'vision':7,'language':7})
        self.assertTrue(all(same['state_dict'][k] is v for k,v in expanded['state_dict'].items()))
        for kernels in ({'vision':3,'language':5},{'vision':8,'language':7},{'vision':7}):
            with self.assertRaises(ValueError):expand_electronic_context(expanded,kernels)

    def test_context7_only_adds_small_existing_electronic_kernels(self):
        from LightGenV2.tasks.t07_abo_image_retrieval.standalone.model import Residual
        old=Residual(True);new=Residual(True,7)
        self.assertEqual(2*(sum(p.numel() for p in new.parameters())-sum(p.numel() for p in old.parameters())),15360)
        self.assertEqual(set(dict(old.named_modules())),set(dict(new.named_modules())))
        wide=overlay_config({'adapt':{},'augmentation':{}},'domain_refine_wide')
        context=overlay_config({'adapt':{},'augmentation':{}},'domain_refine_context7')
        self.assertEqual(context.pop('electronic_context_kernels'),{'vision':7,'language':5})
        wide.pop('protocol');context.pop('protocol')
        self.assertEqual(wide,context)

    def test_restored_proxies_survive_epoch_zero_initialization(self):
        from LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization import initialize_category_proxies
        from torch.nn import functional as F
        head=torch.nn.Linear(3,2,bias=False)
        before=head.weight.detach().clone()
        features=F.normalize(torch.tensor([[1.,2.,3.],[3.,2.,1.],[2.,1.,3.],[3.,1.,2.]]),dim=-1)
        labels=torch.tensor([0,0,1,1])
        result=initialize_category_proxies(head,features,labels,True)
        self.assertEqual(result,'preserved_checkpoint')
        self.assertTrue(torch.equal(head.weight,before))
        result=initialize_category_proxies(head,features,labels)
        expected=torch.stack([F.normalize(features[labels==c].mean(0),dim=0) for c in range(2)])
        self.assertEqual(result,'target_training_class_means')
        self.assertTrue(torch.equal(head.weight,expected))

    def test_full_auxiliary_restore_only_changes_proxy_initialization(self):
        old=overlay_config({'adapt':{},'augmentation':{}},'domain_distill_resumeaux')
        full=overlay_config({'adapt':{},'augmentation':{}},'domain_distill_resumeaux_full')
        self.assertTrue(full.pop('preserve_restored_category_proxies'))
        self.assertEqual(old,full)

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

    def test_both_fullfield_changes_only_vision_vs_language_only_control(self):
        language=overlay_config({'adapt':{},'augmentation':{}},'preserve_fullfield_sam')
        both=overlay_config({'adapt':{},'augmentation':{}},'preserve_fullfield_both_sam')
        self.assertEqual(both['ccd_readout_modes'],{'vision':'fullfield_rows','language':'fullfield_rows'})
        language['ccd_readout_modes']['vision']='fullfield_rows'
        self.assertEqual(language,both)

    def test_fullfield_vision_uses_bottom_without_more_parameters(self):
        optics=OpticalPath()
        names={n:tuple(p.shape) for n,p in optics.named_parameters()}
        raw=torch.zeros(1,478,478);raw[:,450:,170:310]=10
        optics.readout_mode='prefix_rows';old=optics.decode(raw,196,torch.float32,False)
        optics.readout_mode='fullfield_rows';new=optics.decode(raw,196,torch.float32,False)
        self.assertEqual(new.shape,(1,196,192))
        self.assertFalse(torch.allclose(old,new))
        self.assertEqual(names,{n:tuple(p.shape) for n,p in optics.named_parameters()})

    def test_dependency_waits_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'status.json'
            for state,expected in [('running','waiting_for_dependency'),('complete','ready')]:
                path.write_text(json.dumps({'status':state,'gpu':'test-gpu'}))
                self.assertEqual(dependency_state(path,'test-gpu'),expected)
            path.write_text('{')
            self.assertEqual(dependency_state(path,'test-gpu'),'waiting_for_dependency')
            path.write_text(json.dumps({'status':'failed_or_interrupted','gpu':'test-gpu'}))
            with self.assertRaises(RuntimeError):dependency_state(path,'test-gpu')
            path.write_text(json.dumps({'status':'complete','gpu':'someone-else'}))
            with self.assertRaises(RuntimeError):dependency_state(path,'test-gpu')


if __name__ == '__main__':
    unittest.main()
