"""Topology, dark-field gradients, optical kernel and reload contracts."""
import ast
from pathlib import Path
import tempfile
import unittest
import torch
from torch import nn
from ..model import LightGenTemporal, ModelConfig, GlobalStage, Propagation, Router, load_lightgen
from ..run import save_checkpoint
from ..data import attach_train_soft_targets


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
            self.assertNotIsInstance(layer,(nn.GELU,nn.Softplus,nn.Conv1d,nn.Conv2d,nn.MultiheadAttention,nn.Transformer))
        self.assertEqual(sum(isinstance(l,nn.Linear) for l in model.readout.modules()),2)
        names=list(dict(model.named_parameters()))
        self.assertEqual(sum(n.endswith('raw_global_phase') for n in names),2)
        self.assertFalse(any(any(s in n for s in ['quality','fusion','vision_route','language_route','skip']) for n in names))
        source=(Path(__file__).parents[1]/'model.py').read_text('utf-8')
        tree=ast.parse(source)
        imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any(s.startswith(('LightGenV2','experiments','transformers')) for s in imports))

    def test_atomic_checkpoint_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'last_checkpoint.pt'
            for version in [1,2]:
                save_checkpoint(path,{'value':torch.tensor(version)})
                self.assertEqual(int(torch.load(path,weights_only=True)['value']),version)
            self.assertFalse(list(Path(tmp).glob('*.writing')))

    def test_original_merger_structure_and_reduction(self):
        cfg=ModelConfig(width=8,expert_hidden=16,router_hidden=8,readout_hidden=16)
        model=LightGenTemporal(cfg)
        self.assertEqual(model.frame_merger[0].normalized_shape,(16,))
        self.assertEqual((model.frame_merger[1].in_features,model.frame_merger[1].out_features),(16,8))
        self.assertIsInstance(model.frame_merger[2],nn.ReLU)
        self.assertEqual(tuple(model.sequence_position.shape),(1,64,8))
        self.assertTrue(hasattr(model,'visual_input_norm'))
        self.assertFalse(hasattr(model,'prompt_to_visual'))

    def test_visual_stage_is_independent_of_prompt(self):
        cfg=ModelConfig(width=8,expert_hidden=16,router_hidden=8,readout_hidden=16,
                        input_shift=0,phase_shift=0,ccd_shift=0,phase_dropout=0)
        model=LightGenTemporal(cfg).eval()
        pixels=torch.randn(1,16,4,49,1024)
        with torch.inference_mode():
            model.set_prompt(torch.randn(38,2048))
            before=model.encode_frames(pixels)
            model.set_prompt(torch.randn(38,2048)*100)
            after=model.encode_frames(pixels)
        self.assertTrue(torch.equal(before['frames'],after['frames']))
        self.assertTrue(torch.equal(before['routing']['weights'],after['routing']['weights']))

    def test_teacher_matches_train_ids_and_rejects_test_entries(self):
        payload={'ids':['a','b','c'],'splits':['train','test','train'],'identity':{}}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'teacher.pt'
            teacher={'sample_ids':['c','a'],'predictions':torch.tensor([[1.,30.],[2.,50.]]),
                     'target_names':['spatial','temporal'],'inference_dependency':False}
            torch.save(teacher,path);attach_train_soft_targets(payload,path)
            self.assertEqual(payload['teacher'].tolist(),[50.,0.,30.])
            self.assertEqual(payload['teacher_present'].tolist(),[True,False,True])
            teacher['sample_ids']=['a','b'];torch.save(teacher,path)
            with self.assertRaises(ValueError):attach_train_soft_targets(payload,path)

    def test_legacy_gelu_checkpoint_preserves_its_predictions(self):
        config = ModelConfig(width=8, expert_hidden=16, router_hidden=8,
                             readout_hidden=16, mlp_activation='gelu',
                             input_shift=0, phase_shift=0, ccd_shift=0,
                             phase_dropout=0)
        model = LightGenTemporal(config).eval()
        model.set_prompt(torch.randn(38, 2048))
        inputs = torch.randn(1, 16, 4, 49, 1024)
        with torch.inference_mode():
            expected = model(inputs)['prediction']
        legacy = model.checkpoint()
        legacy['config'].pop('mlp_activation')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'legacy.pt'
            torch.save(legacy, path)
            restored, _ = load_lightgen(path)
            self.assertEqual(restored.cfg.mlp_activation, 'gelu')
            with torch.inference_mode():
                actual = restored(inputs)['prediction']
            self.assertTrue(torch.equal(expected, actual))


if __name__=='__main__':unittest.main()
