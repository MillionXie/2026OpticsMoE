import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from lightgen_abo.model import alpha_value, fuse, OpticalRetrieval
from lightgen_abo.objectives import product_bank, gallery_loss, phase_change


class Contracts(unittest.TestCase):
    def test_no_repository_dependencies(self):
        root = Path(__file__).resolve().parents[1] / 'lightgen_abo'
        for source in root.glob('*.py'):
            tree = ast.parse(source.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or '']
                else:
                    continue
                self.assertFalse(any(n.startswith(('LightGenV2', 'experiments', 'qwen_vl_utils')) for n in names), source)
            self.assertNotIn('AutoModel', source.read_text())

    def test_alpha_and_scale(self):
        bounds = (.4001, .8)
        self.assertGreater(float(alpha_value(torch.tensor(-100.), bounds)), .4)
        e = torch.randn(2, 77, 192)
        o = torch.randn_like(e)
        a = torch.tensor(0., requires_grad=True)
        z = fuse(e, o, a, bounds)
        self.assertEqual(z.shape, e.shape)
        self.assertTrue(torch.allclose(z.square().mean((1, 2)), e.square().mean((1, 2)), atol=1e-5))
        z.sum().backward()
        self.assertIsNotNone(a.grad)

    def test_train_gallery_excludes_own_product(self):
        samples = [SimpleNamespace(product_id=str(i), category_id=i//2, split='train') for i in range(4)]
        f = torch.tensor([[1., 0.], [.9, .1], [0., 1.], [.1, .9]])
        bank, labels, own = product_bank(f, samples)
        nll, margin, hit = gallery_loss(f.requires_grad_(), labels, own, bank, labels)
        self.assertAlmostEqual(float(hit), 1.)
        (nll + margin).backward()
        self.assertTrue(torch.isfinite(f.grad).all())

    def test_model_audit_and_phase_gradient(self):
        torch.set_num_threads(2)
        model = OpticalRetrieval(dict(token_count=22, input_rms=.5, fusion_alpha_min=.4001, fusion_alpha_max=.8))
        audit = model.audit()
        self.assertEqual(audit['capture_count'], 6)
        self.assertEqual(audit['trainable_parameters'], 2782485)
        self.assertEqual(audit['attention_modules'], 0)
        model.vision.optics.eval()
        latent = torch.randn(1, 77, 192)
        z, _ = model.vision.optics.expert(latent)
        z.square().mean().backward()
        grads = [p.grad for p in model.vision.optics.experts]
        self.assertTrue(any(g is not None and bool(g.abs().sum() > 0) for g in grads))


if __name__ == '__main__':
    unittest.main()
