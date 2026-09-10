import ast
from pathlib import Path
import sys
import unittest
import torch

TASK=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(TASK))
from standalone.model import Residual, Modality, fuse, alpha_value, rms
from standalone.optics import Router, OpticalPath
from standalone.data import _ranking_metrics
from standalone.curriculum import stage_settings, parameter_kind, relation_loss


class StandaloneTests(unittest.TestCase):
    def test_training_gallery_excludes_own_product_and_is_detached(self):
        import types
        from standalone.retrieval_training import product_bank,gallery_loss
        samples=[types.SimpleNamespace(product_id=f'p{i//2}',category_id=i//4,split='train') for i in range(8)]
        features=torch.tensor([[1.,0.],[1.,0.],[.9,.1],[1.,0.],[0.,1.],[0.,1.],[.1,.9],[0.,1.]],requires_grad=True)
        bank,labels,ids=product_bank(features,samples)
        self.assertFalse(bank.requires_grad);self.assertEqual(bank.shape,(4,2))
        query=torch.tensor([[.9,.1]],requires_grad=True)
        nll,margin,hit=gallery_loss(query,torch.tensor([0]),ids[:1],bank,labels)
        altered=bank.clone();altered[0]=torch.tensor([-1.,0.])
        nll2,margin2,_=gallery_loss(query,torch.tensor([0]),ids[:1],altered,labels)
        torch.testing.assert_close(nll,nll2);torch.testing.assert_close(margin,margin2)
        (nll+margin).backward();self.assertTrue(torch.isfinite(query.grad).all());self.assertIsNone(features.grad)
        self.assertEqual(float(hit),1.)
        samples[0].split='test'
        with self.assertRaises(ValueError):product_bank(features,samples)

    def test_retrieval_loss_prefers_positive_and_polish_schedule(self):
        from standalone.retrieval_training import gallery_loss,readout_polish
        bank=torch.tensor([[1.,0.],[.9,.1],[0.,1.],[.1,.9]])
        labels=torch.tensor([0,0,1,1]);own=torch.tensor([0]);target=torch.tensor([0])
        good=gallery_loss(torch.tensor([[1.,0.]]),target,own,bank,labels)
        bad=gallery_loss(torch.tensor([[0.,1.]]),target,own,bank,labels)
        self.assertLess(float(good[0]+good[1]),float(bad[0]+bad[1]))
        cfg={'readout_polish_epochs':5}
        self.assertFalse(readout_polish(25,30,cfg));self.assertTrue(readout_polish(26,30,cfg))
        self.assertFalse(readout_polish(1,1,cfg))

    def test_legacy_fusion_is_unchanged_and_high_alpha_bounded(self):
        e,o=torch.randn(2,7,192),torch.randn(2,7,192);raw=torch.tensor(-2.)
        re,ro=rms(e).detach(),rms(o).detach();a=.01+.94*raw.sigmoid()
        mixture=(1-a)*e/re+a*o/ro
        # Exact same parenthesization as the original implementation.
        mixture=(1-a)*(e/re)+a*(o/ro)
        self.assertTrue(torch.equal(fuse(e,o,raw),re*mixture/rms(mixture).detach()))
        values=alpha_value(torch.tensor([-100.,0.,100.]),(.4,.8))
        self.assertTrue(bool(((values>=.4)&(values<=.8)).all()))

    def test_high_alpha_upgrade_and_auxiliary_gradients(self):
        import json,types
        from standalone.high_alpha import convert_payload,optical_heads,optical_classification_loss
        cfg=json.loads((TASK/'standalone/high_alpha.json').read_text())
        name='vision.block1_optical_fusion_logit'
        p=convert_payload({'metadata':{},'state_dict':{name:torch.tensor(-3.)}},cfg)
        bounds=(cfg['fusion_alpha_min'],cfg['fusion_alpha_max'])
        self.assertAlmostEqual(float(alpha_value(p['state_dict'][name],bounds)),.45,places=6)
        self.assertGreater(float(alpha_value(torch.tensor(-1000.),bounds)),.4)
        p['state_dict'][name]=torch.tensor(0.)
        self.assertEqual(float(convert_payload(p,cfg)['state_dict'][name]),0.)
        v,l=[torch.randn(4,7,192,requires_grad=True) for _ in range(2)]
        m=types.SimpleNamespace(vision=types.SimpleNamespace(last_optical=v),language=types.SimpleNamespace(last_optical=l))
        optical_classification_loss(m,optical_heads(10),torch.arange(4)).backward()
        self.assertGreater(float(v.grad.abs().sum()),0);self.assertGreater(float(l.grad.abs().sum()),0)

    def test_broad_batch_and_training_only_head(self):
        import random
        from standalone.broad_transfer import CategoryProxies,sampled_indices
        groups={c:{f'p{c}_{p}':[c*100+p] for p in range(6)} for c in range(12)}
        batch=sampled_indices(groups,8,4,random.Random(42))
        self.assertEqual(len(batch),32);self.assertEqual(len(set(batch)),32)
        from collections import Counter
        self.assertEqual(set(Counter(i//100 for i in batch).values()),{4})
        head=CategoryProxies(128);z=torch.randn(32,64,requires_grad=True)
        head(z).square().mean().backward()
        self.assertEqual(head(z).shape,(32,128));self.assertTrue(torch.isfinite(z.grad).all())

    def test_pretraining_duplicate_screen_and_path(self):
        import numpy as np
        from standalone.prepare_broad_abo import near_duplicate,safe_image
        protected=np.zeros((2,16),dtype=np.uint8)
        self.assertTrue(near_duplicate(np.zeros(16,dtype=np.uint8),protected))
        self.assertFalse(near_duplicate(np.full(16,255,dtype=np.uint8),protected))
        with self.assertRaises(ValueError):safe_image(TASK,'../../outside.jpg')

    def test_curriculum_stages_and_groups(self):
        import json
        cfg=json.loads((TASK/'standalone/curriculum.json').read_text())
        self.assertEqual(stage_settings(1,30,cfg)['phase_scale'],0.)
        self.assertEqual(stage_settings(5,30,cfg)['name'],'joint')
        self.assertEqual(stage_settings(30,30,cfg)['teacher'],0.)
        self.assertEqual(parameter_kind('vision.block1_optical_fusion_logit'),'alpha')
        self.assertEqual(parameter_kind('language.optics.experts.0'),'phase')
        self.assertEqual(parameter_kind('vision.optics.router.raw_router_phase'),'router')

    def test_relation_teacher_detached(self):
        x=torch.randn(10,64,requires_grad=True);t=torch.randn(10,64,requires_grad=True)
        loss=relation_loss(x,t);loss.backward()
        self.assertIsNotNone(x.grad);self.assertIsNone(t.grad)
        self.assertTrue(torch.isfinite(loss))

    def test_no_external_repository_imports(self):
        for path in (TASK/'standalone').glob('*.py'):
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                if isinstance(node,ast.ImportFrom):
                    self.assertFalse((node.module or '').startswith(('experiments','LightGenV2')))
                if isinstance(node,ast.Import):
                    self.assertFalse(any(n.name.startswith(('experiments','LightGenV2')) for n in node.names))

    def test_fusion_preserves_rms(self):
        e,o=torch.randn(2,8,192),torch.randn(2,8,192)
        f=fuse(e,o,torch.tensor(-2.))
        torch.testing.assert_close(f.square().mean((1,2)),e.square().mean((1,2)))

    def test_residual_shapes_and_gradients(self):
        for vision,length in [(True,196),(False,83)]:
            net=Residual(vision).eval()
            x=torch.randn(1,length,192,requires_grad=True)
            net(x).square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertIsNotNone(net.token_depthwise.weight.grad)

    def test_optical_geometry_and_router(self):
        torch.set_num_threads(2)
        path=OpticalPath(.5).eval()
        field=path.encode(torch.randn(2,7,192))
        self.assertEqual(field.shape,(2,224,224))
        self.assertTrue(torch.all(field[:,7:]==0))
        weights=path.router(field)
        self.assertTrue(torch.all(path.router.last['selected_mask'].sum(1)==2))
        torch.testing.assert_close(weights.square().sum(1),torch.ones(2))
        canvas=path.fanout(field,weights)
        self.assertEqual(canvas.shape,(2,518,518))
        self.assertTrue(torch.all(canvas[:,:20]==0))
        self.assertTrue(torch.all(canvas[:,244:274]==0))

    def test_hit_not_positive_recall(self):
        import numpy as np
        r=_ranking_metrics(np.ones((2,12),dtype=bool))
        self.assertEqual(r['hit_at_1'],1.)
        self.assertAlmostEqual(r['positive_recall_at_1'],1/12)


if __name__=='__main__':unittest.main()
