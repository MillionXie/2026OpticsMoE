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


class StandaloneTests(unittest.TestCase):
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
