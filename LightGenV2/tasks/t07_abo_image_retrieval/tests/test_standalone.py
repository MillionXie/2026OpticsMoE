import ast
from pathlib import Path
import sys
import unittest
import torch

TASK=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(TASK))
from standalone.model import Residual, Modality, fuse
from standalone.optics import Router, OpticalPath
from standalone.data import _ranking_metrics
from standalone.curriculum import stage_settings, parameter_kind, relation_loss


class StandaloneTests(unittest.TestCase):
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
