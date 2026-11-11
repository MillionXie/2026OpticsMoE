"""Topology, dark-field gradients, optical kernel and reload contracts."""
import ast
from pathlib import Path
import tempfile
import unittest
import torch
from torch import nn
from ..model import LightGenTemporal, ModelConfig, GlobalStage, Propagation, Router, load_lightgen


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(4)

    def test_router_exactly_top2_and_l1(self):
        cfg=ModelConfig(width=8,router_hidden=8)
        r=Router(cfg,49).eval()(torch.randn(2,16,4,49,8))
        self.assertTrue(torch.all((r['weights']>0).sum(-1)==2))
        self.assertTrue(torch.allclose(r['weights'].sum(-1),torch.ones(2,16,4)))

    def test_dark_relu_field_has_finite_backward(self):
        cfg=ModelConfig(width=8,input_shift=0,phase_shift=0,ccd_shift=0,phase_dropout=0)
        stage=GlobalStage(cfg,True).eval()
        for p in [stage.width_to_field.weight,stage.tokens_to_field.weight]:nn.init.zeros_(p)
        for p in [stage.width_to_field.bias,stage.tokens_to_field.bias]:nn.init.constant_(p,-1)
        x=torch.randn(1,16,4,49,8,requires_grad=True)
        result,_=stage(x);result.sum().backward()
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_serial_forward_backward_and_reload(self):
        cfg=ModelConfig(width=8,expert_hidden=16,router_hidden=8,readout_hidden=16,
                        input_shift=0,phase_shift=0,ccd_shift=0,phase_dropout=0)
        model=LightGenTemporal(cfg).eval();model.set_prompt(torch.randn(38,2048))
        x=torch.randn(1,16,4,49,1024)
        out=model(x);self.assertEqual(tuple(out['prediction'].shape),(1,16))
        out['prediction'].square().mean().backward()
        for name,param in model.named_parameters():
            self.assertIsNotNone(param.grad,name)
            self.assertTrue(torch.isfinite(param.grad).all(),name)
        # Prove that MOS actually depends on both retained global phase planes.
        for stage in [model.frame_global,model.video_global]:
            self.assertGreater(float(stage.raw_global_phase.grad.abs().sum()),0)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'LightGen.pt';torch.save(model.checkpoint(test=True),path)
            again,meta=load_lightgen(path)
            self.assertTrue(torch.equal(out['prediction'],again(x)['prediction']))

    def test_no_disallowed_layers_or_shortcuts(self):
        model=LightGenTemporal(ModelConfig(width=8,expert_hidden=16,router_hidden=8,readout_hidden=16))
        for layer in model.modules():
            self.assertNotIsInstance(layer,(nn.Softplus,nn.Conv1d,nn.Conv2d,nn.MultiheadAttention,nn.Transformer))
        self.assertEqual(sum(isinstance(l,nn.Linear) for l in model.readout.modules()),2)
        names=list(dict(model.named_parameters()))
        self.assertEqual(sum(n.endswith('raw_global_phase') for n in names),2)
        self.assertFalse(any(any(s in n for s in ['quality','fusion','vision_route','language_route','skip']) for n in names))
        source=(Path(__file__).parents[1]/'model.py').read_text('utf-8')
        tree=ast.parse(source)
        imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any(s.startswith(('LightGenV2','experiments','transformers')) for s in imports))


if __name__=='__main__':unittest.main()
