"""CPU-only auxiliary loss contract test; no dataset, SDK or CUDA allocation."""
import ast
from pathlib import Path
import unittest

import torch


class PairingTest(unittest.TestCase):
    def test_observed_response_finite_and_identity(self):
        source=Path(__file__).with_name('train_editor16_robust_chain.py')
        tree=ast.parse(source.read_text(encoding='utf-8'))
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='observed_response_ccd')
        scope={'torch':torch}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),scope)
        stats={'mean':.2,'p01':.04}
        for value in (0.,.5):
            x=torch.full((2,32,32),value,requires_grad=True)
            self.assertIs(scope['observed_response_ccd'](x,0,stats),x)
            y=scope['observed_response_ccd'](x,1,stats)
            self.assertTrue(torch.isfinite(y).all())
            y.sum().backward()
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_finite_and_teacher_detached(self):
        source=Path(__file__).with_name('train_editor16_robust_chain.py')
        tree=ast.parse(source.read_text(encoding='utf-8'))
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='paired_consistency')
        scope={'torch':torch}
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),scope)
        clean={k:torch.randn(shape,requires_grad=True) for k,shape in
               [('category_logits',(2,5,4,4)),('edit_logits',(2,1,4,4))]}
        noisy={k:torch.randn_like(v,requires_grad=True) for k,v in clean.items()}
        loss=scope['paired_consistency'](noisy,clean)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(v.grad is None for v in clean.values()))
        self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in noisy.values()))
        equal=scope['paired_consistency'](clean,clean)
        self.assertLess(abs(float(equal.detach())),1e-6)


if __name__=='__main__':
    unittest.main()
