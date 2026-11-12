"""Topology, dark-field gradients, optical kernel and reload contracts."""
import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import torch
from torch import nn
from ..model import LightGenTemporal, ModelConfig, GlobalStage, Propagation, Router, Experts, load_lightgen
from ..run import save_checkpoint, selection_score, router_regularization
from ..data import attach_train_soft_targets
from ..infer import frames_from_video, iter_video_fields
from ..phase_regularization import phase_penalties, gaussian_phase, illumination_support, initialize_phase
from ..training_policy import freeze_parameters,restrict_training_to
from ..prepare_joint_router import expand_router
from ..phase_plotting import phase_display_norm


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(4)

    def test_router_exactly_top2_and_l1(self):
        cfg=ModelConfig(width=8,router_hidden=8)
        r=Router(cfg,49).eval()(torch.randn(2,16,4,49,8))
        self.assertTrue(torch.all((r['weights']>0).sum(-1)==2))
        self.assertTrue(torch.allclose(r['weights'].sum(-1),torch.ones(2,16,4)))

    def test_all_four_softmax_then_selected_pair_renormalization(self):
        cfg=ModelConfig(width=8,router_hidden=8)
        router=Router(cfg,49).eval()
        expected=torch.tensor([.1,.2,.3,.4])
        with torch.no_grad():
            router.fc1.weight.zero_();router.fc1.bias.zero_()
            router.fc2.bias.copy_(expected.log()*cfg.temperature)
        result=router(torch.randn(1,49,8))
        self.assertTrue(torch.allclose(result['probabilities'][0],expected,atol=1e-6))
        self.assertTrue(torch.allclose(result['weights'][0],torch.tensor([0.,0.,3/7,4/7]),atol=1e-6))

    def test_balance_selection_preserves_accuracy_floor_and_default(self):
        metrics={'srcc':.79,'frame_routing':{'selection_share':[.4,.1,.1,.4]}}
        self.assertEqual(selection_score(metrics,{}),.79)
        config={'selection_frame_share_weight':.05,'reference_srcc':.8,
                'allowed_absolute_srcc_difference':.02}
        balanced={'srcc':.789,'frame_routing':{'selection_share':[.25]*4}}
        self.assertGreater(selection_score(balanced,config),selection_score(metrics,config))
        self.assertEqual(selection_score({**balanced,'srcc':.77},config),-float('inf'))

    def test_frame_regularization_scale_does_not_change_video_term(self):
        cfg=ModelConfig(width=8,router_hidden=8,router_noise=0)
        routes={'frame':Router(cfg,49).eval()(torch.randn(3,16,4,49,8)),
                'video':Router(cfg,4).eval()(torch.randn(3,16,4,8))}
        valid=torch.ones(3,16,dtype=torch.bool)
        combined,_=router_regularization(routes,valid)
        scaled,_=router_regularization(routes,valid,frame_scale=3)
        visual,_=router_regularization({'frame':routes['frame']},valid)
        for name in combined:
            self.assertTrue(torch.allclose(scaled[name]-combined[name],visual[name],atol=1e-6))

    def test_training_freeze_is_checked_and_preserves_forward(self):
        cfg=ModelConfig(width=8,router_hidden=8,router_noise=0)
        model=LightGenTemporal(cfg).eval()
        value=torch.randn(1,16,4,49,1024)
        before=model.frame_router(model.vision_adapter(value))['weights']
        frozen=freeze_parameters(model,['vision_adapter','frame_router'])
        self.assertTrue(frozen)
        self.assertTrue(all(not p.requires_grad for n,p in model.named_parameters() if n in frozen))
        after=model.frame_router(model.vision_adapter(value))['weights']
        self.assertTrue(torch.equal(before,after))
        with self.assertRaises(ValueError):freeze_parameters(model,['missing_module'])

    def test_joint_router_expansion_preserves_fixed_prompt_and_responds_to_text(self):
        cfg=ModelConfig(width=8,expert_mode='linear',orm_mode='amplitude',
                        router_hidden=8,readout_hidden=16,input_shift=0,phase_shift=0,
                        ccd_shift=0,phase_dropout=0,router_noise=0)
        parent=LightGenTemporal(cfg).eval()
        parent.set_prompt(torch.randn(38,2048))
        parent.set_quality_levels(torch.tensor([0.,100.]))
        model=expand_router(parent,277)
        self.assertEqual(model.video_router.fc1.in_features,42*8)
        value=torch.randn(1,16,4,49,1024)
        with torch.no_grad():
            before=parent(value)
            after=model(value)
            self.assertTrue(torch.allclose(before['prediction'],after['prediction'],atol=2e-5))
            model.set_prompt(model.prompt_embedding.roll(13,-1))
            changed=model(value)
        self.assertTrue(torch.equal(after['routing']['frame']['weights'],changed['routing']['frame']['weights']))
        self.assertGreater(float((after['routing']['video']['probabilities']-
                                  changed['routing']['video']['probabilities']).abs().max()),1e-6)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'joint.pt'
            torch.save(model.checkpoint(),path)
            restored,_=load_lightgen(path)
            self.assertEqual(restored.cfg.fusion_router_input,'all_tokens')
            self.assertEqual(restored.checkpoint()['schema'],7)
            with torch.no_grad():
                self.assertTrue(torch.equal(changed['prediction'],restored(value)['prediction']))

    def test_trainable_whitelist_freezes_all_other_parameters(self):
        model=LightGenTemporal(ModelConfig(width=8,readout_hidden=16))
        frozen=restrict_training_to(model,['readout','video_global.raw_global_phase'])
        self.assertTrue(frozen)
        for name,value in model.named_parameters():
            expected=name.startswith('readout.') or name=='video_global.raw_global_phase'
            self.assertEqual(value.requires_grad,expected,name)

    def test_nonlinear_phase_color_mapping_is_monotone_and_invertible(self):
        import numpy as np
        values=np.linspace(0,2*np.pi,101)
        norm=phase_display_norm()
        mapped=norm(values)
        self.assertTrue(np.all(np.diff(mapped)>0))
        self.assertTrue(np.allclose(norm.inverse(mapped),values,atol=1e-9))

    def test_native_tail_padding_matches_global_input_order(self):
        def decode(index):
            return torch.full((4,3,2,2), index, dtype=torch.int64)
        with patch('LightGenV2.tasks.t22_public_temporal_mlp.infer.frames_from_video', side_effect=decode):
            for count in (1,3,15,16,17,31,32,33,558):
                fields = list(iter_video_fields(list(range(count))))
                actual = torch.cat([pixels[0,:,0,0,0,0] for _,pixels in fields]).tolist()
                padding = (-count) % 16
                expected = list(range(count)) + [i % count for i in range(padding)]
                self.assertEqual(actual, expected, count)

    def test_unreadable_video_releases_decoder(self):
        decoder = Mock()
        decoder.get.return_value = 0
        cv2 = Mock()
        cv2.VideoCapture.return_value = decoder
        with patch.dict('sys.modules', {'cv2': cv2}):
            with self.assertRaises(ValueError):
                frames_from_video('unreadable.mp4')
        decoder.release.assert_called_once_with()

    def test_periodic_phase_regularization_and_support(self):
        phase = torch.randn(2,111,111,requires_grad=True)
        original = phase_penalties(phase)
        shifted = phase_penalties(phase + 2*torch.pi)
        self.assertTrue(torch.allclose(original['curvature'],shifted['curvature'],atol=1e-6))
        smoothed = gaussian_phase(phase.detach(),1.5)
        self.assertLess(float(phase_penalties(smoothed)['curvature']),float(original['curvature'].detach()))
        original['curvature'].backward()
        self.assertTrue(torch.isfinite(phase.grad).all())
        self.assertEqual(int(illumination_support(True).sum()),4*27*27)
        self.assertEqual(int(illumination_support(False).sum()),42*56)

    def test_phase_noise_is_training_only(self):
        cfg = ModelConfig(width=8,orm_mode='amplitude',phase_noise_std=.02,
                          phase_shift=0,phase_dropout=0)
        stage = GlobalStage(cfg,False)
        stage.eval()
        self.assertTrue(torch.equal(stage.modulation(),stage.modulation()))
        stage.train()
        self.assertFalse(torch.equal(stage.modulation(),stage.modulation()))

    def test_phase_contrast_initialization_uses_existing_smooth_structure(self):
        phase=2+.01*torch.randn(2,111,111)
        unchanged=initialize_phase(phase,{'kind':'contrast','sigma_pixels':0,'deviation_scale':1},ModelConfig())
        amplified=initialize_phase(phase,{'kind':'contrast','sigma_pixels':0,'deviation_scale':2},ModelConfig())
        self.assertTrue(torch.allclose(unchanged,phase,atol=1e-6))
        self.assertTrue(torch.isfinite(amplified).all())
        self.assertGreater(float(amplified.std()),float(phase.std()))

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
        model.set_quality_levels(torch.tensor([0., 100.]))
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
                             readout_mode='scalar', visual_output_norm=True,
                             input_shift=0, phase_shift=0, ccd_shift=0,
                             phase_dropout=0)
        model = LightGenTemporal(config).eval()
        model.set_prompt(torch.randn(38, 2048))
        inputs = torch.randn(1, 16, 4, 49, 1024)
        with torch.inference_mode():
            expected = model(inputs)['prediction']
        legacy = model.checkpoint()
        legacy['config'].pop('mlp_activation')
        legacy['config'].pop('readout_mode')
        legacy['config'].pop('visual_output_norm')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'legacy.pt'
            torch.save(legacy, path)
            restored, _ = load_lightgen(path)
            self.assertEqual(restored.cfg.mlp_activation, 'gelu')
            with torch.inference_mode():
                actual = restored(inputs)['prediction']
            self.assertTrue(torch.equal(expected, actual))

    def test_five_level_expectation_and_projection_norm_switch(self):
        config = ModelConfig(width=8, expert_hidden=16, router_hidden=8,
                             readout_hidden=16, input_shift=0, phase_shift=0,
                             ccd_shift=0, phase_dropout=0)
        model = LightGenTemporal(config).eval()
        model.set_prompt(torch.randn(38, 2048))
        model.set_quality_levels(torch.tensor([0., 100.]))
        self.assertIsInstance(model.visual_input_norm, nn.Identity)
        with torch.no_grad():
            model.readout[3].weight.zero_()
            model.readout[3].bias.copy_(torch.tensor([-2., -1., 0., 1., 2.]))
            model.target_mean.fill_(1000.)
            model.target_std.fill_(77.)
            result = model(torch.randn(1, 16, 4, 49, 1024))
        probabilities = torch.tensor([-2., -1., 0., 1., 2.]).softmax(0)
        expected = probabilities @ torch.tensor([0., 25., 50., 75., 100.])
        self.assertTrue(torch.allclose(result['prediction'], expected.expand(1, 16)))
        self.assertTrue(torch.allclose(result['quality_probabilities'].sum(-1), torch.ones(1, 16)))
        self.assertEqual(tuple(result['quality_logits'].shape), (1, 16, 5))
        with_norm = LightGenTemporal(ModelConfig(width=8, expert_hidden=16,
            router_hidden=8, readout_hidden=16, visual_output_norm=True))
        self.assertIsInstance(with_norm.visual_input_norm, nn.LayerNorm)

    def test_linear_experts_have_no_bias_norm_or_activation(self):
        cfg = ModelConfig(width=8, expert_mode='linear', orm_mode='amplitude')
        experts = Experts(cfg).eval()
        x, y = torch.randn(3, 49, 8), torch.randn(3, 49, 8)
        for expert in experts.experts:
            self.assertIsInstance(expert, nn.Linear)
            self.assertIsNone(expert.bias)
            self.assertTrue(torch.allclose(expert(2*x + 3*y), 2*expert(x) + 3*expert(y), atol=2e-6))

    def test_manuscript_amplitude_orm(self):
        cfg = ModelConfig(width=8, orm_mode='amplitude', alpha_eval=2/3,
                          phase_shift=0, phase_dropout=0)
        stage = GlobalStage(cfg, False).eval()
        with torch.no_grad():
            stage.raw_global_phase.zero_()  # D = exp(i*pi) = -1.
        expected = torch.full((16,111,111), -1/3, dtype=torch.complex64)
        self.assertTrue(torch.allclose(stage.modulation(), expected, atol=1e-6))

    def test_amplitude_linear_checkpoint_roundtrip_and_legacy_config(self):
        cfg = ModelConfig(width=8, expert_hidden=16, router_hidden=8,
                          readout_hidden=16, expert_mode='linear', orm_mode='amplitude',
                          input_shift=0, phase_shift=0, ccd_shift=0, phase_dropout=0)
        model = LightGenTemporal(cfg).eval()
        model.set_prompt(torch.randn(38,2048))
        model.set_quality_levels(torch.tensor([0.,100.]))
        x = torch.randn(1,16,4,49,1024)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'linear.pt'
            torch.save(model.checkpoint(), path)
            restored, _ = load_lightgen(path)
            self.assertEqual(restored.cfg, cfg)
            with torch.inference_mode():
                self.assertTrue(torch.equal(model(x)['prediction'], restored(x)['prediction']))
            old = LightGenTemporal(ModelConfig(width=8, expert_hidden=16,
                router_hidden=8, readout_hidden=16)).checkpoint()
            for key in ('expert_mode','orm_mode','alpha_min','alpha_max','alpha_eval'):
                old['config'].pop(key)
            torch.save(old, path)
            restored, _ = load_lightgen(path)
            self.assertEqual(restored.cfg.orm_mode, 'legacy_energy')


if __name__=='__main__':unittest.main()
