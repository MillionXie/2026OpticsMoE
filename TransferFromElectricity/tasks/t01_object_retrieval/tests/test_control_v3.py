import tempfile
import json
import unittest
from pathlib import Path
import torch
from TransferFromElectricity.tasks.t01_object_retrieval.models.generator import StaticGenerator, install_lora
from TransferFromElectricity.tasks.t01_object_retrieval.datasets import prepare_imagenette, IMAGENETTE_CLASSES
from TransferFromElectricity.tasks.t01_object_retrieval.protocol import split_train_validation


class ControlTests(unittest.TestCase):
    def test_deterministic_pool_adjoint_matches_native(self):
        from TransferFromElectricity.tasks.t01_object_retrieval import deterministic_ops as ops
        for shape, output in [((2,3,7),4), ((2,3,8),4), ((2,3,3),5),
                              ((2,3,7,9),(4,5)), ((2,3,8,8),(4,4)), ((3,3,4),(5,6))]:
            dimensions = 1 if isinstance(output,int) else 2
            native = ops._pool1d if dimensions == 1 else ops._pool2d
            custom = ops.adaptive_avg_pool1d if dimensions == 1 else ops.adaptive_avg_pool2d
            x=torch.randn(shape,dtype=torch.float64,requires_grad=True)
            a,b=native(x,output),custom(x,output)
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            gradient=torch.randn_like(a)
            ga=torch.autograd.grad(a,x,gradient)[0];gb=torch.autograd.grad(b,x,gradient)[0]
            torch.testing.assert_close(ga,gb,rtol=1e-12,atol=1e-12)

    def test_development_selection_uses_complete_validation_grid(self):
        from TransferFromElectricity.tasks.t01_object_retrieval.report_control import development
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);runs=[]
            for i,(e,g) in enumerate(( (e,g) for e in (1e-4,1e-5) for g in (1e-4,1e-3) )):
                path=root/str(i);path.mkdir();runs.append(path)
                cfg={'method':'qwen_lora','learning_rates':{'electronic':e,'readout':e,'generator_context':g}}
                metric={'top1_retrieval_accuracy':.2+i*.1,'top3_retrieval_accuracy':.7,'mrr':.5}
                result={'evaluation_split':'validation','selected_live_test':None,'selected_metrics':metric,
                        'export_max_error':0,'git_sha':'same','split_sha256':'same',
                        'selected_expert_phase':{'rms_change_rad':.5},'ablations':{'lora_phase_effect':{'rms_change_rad':.2}}}
                history=[{'frozen_parameter_max_change':0,'live_validation':metric}]*3
                for name,value in {'protocol':cfg,'final_report':result,'environment':{'device':'RTX test','deterministic_algorithms':True},
                                   'history':history,'status':{'status':'complete'}}.items():
                    (path/(name+'.json')).write_text(json.dumps(value))
            self.assertEqual(development(runs)['chosen']['run_id'],'3')
            with self.assertRaisesRegex(ValueError,'complete paired'):development(runs[:-1])
            result['evaluation_split']='test'
            (runs[-1]/'final_report.json').write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError,'must not inspect test'):development(runs)

    def test_frozen_decoder_is_preserved_in_checkpoint(self):
        generator = StaticGenerator(size=8, patch=4, context_width=32, expert_specific_heads=True)
        with torch.no_grad(): generator.decoder.positions.add_(.3)
        generator.decoder.requires_grad_(False)
        state = generator.compact_state()
        self.assertIn('decoder.positions', state)
        restored = StaticGenerator(size=8, patch=4, context_width=32, expert_specific_heads=True)
        restored.load_compact_state(state)
        torch.testing.assert_close(generator(), restored())

    def test_clip_lora_task_gradient_reaches_adapter(self):
        from transformers import CLIPTextConfig, CLIPTextModel
        model = CLIPTextModel(CLIPTextConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=1, num_attention_heads=4, max_position_embeddings=8,
                            bos_token_id=30, eos_token_id=31, pad_token_id=0))
        model.requires_grad_(False)
        names = install_lora(model, 2)
        self.assertEqual(len(names), 2)
        output = model(input_ids=torch.tensor([[30, 4, 9, 31, 0]])).pooler_output
        (output * torch.arange(16)).sum().backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for n,p in model.named_parameters() if n.endswith('lora_b')), 0)
        self.assertTrue(all(p.grad is None for n,p in model.named_parameters() if '.base.' in n))

    def test_imagenette_official_val_remains_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ('train', 'val'):
                for name in IMAGENETTE_CLASSES:
                    folder = root / 'images' / split / name
                    folder.mkdir(parents=True)
                    for i in range(100): (folder / f'{i:04d}.JPEG').touch()
            bundle = prepare_imagenette({'root': 'images', 'gallery_per_class': 5}, root, root, 42)
            training, validation = split_train_validation(bundle.train_samples, 42, 20)
            sets = [set(s.sample_id for s in samples) for samples in (training,validation,bundle.gallery_samples,bundle.test_samples)]
            self.assertEqual([len(s) for s in sets], [750,200,50,1000])
            self.assertTrue(all(not sets[i] & sets[j] for i in range(4) for j in range(i)))
            self.assertTrue(all(':val:' in sample for sample in sets[3]))
