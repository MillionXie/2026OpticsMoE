import tempfile
import unittest
from pathlib import Path
import torch
from TransferFromElectricity.tasks.t01_object_retrieval.models.generator import StaticGenerator, install_lora
from TransferFromElectricity.tasks.t01_object_retrieval.datasets import prepare_imagenette, IMAGENETTE_CLASSES
from TransferFromElectricity.tasks.t01_object_retrieval.protocol import split_train_validation


class ControlTests(unittest.TestCase):
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
