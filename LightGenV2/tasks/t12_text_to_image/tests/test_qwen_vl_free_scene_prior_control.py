"""CPU-only prompt/seed/native-CLIP contracts for the public-prior control."""
from collections import Counter
from types import SimpleNamespace
import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest

import torch

from ..qwen_vl_free_scene_prior_control import (control_prompts, control_records,
    local_component_manifest, native_text_conditions)


def local_fixture(directory):
    files = {}
    for component in ('unet', 'vae', 'text_encoder'):
        folder = directory/component; folder.mkdir()
        safe = 'model.fp16.safetensors' if component == 'text_encoder' else 'diffusion_pytorch_model.fp16.safetensors'
        binary = 'pytorch_model.fp16.bin' if component == 'text_encoder' else 'diffusion_pytorch_model.fp16.bin'
        for name in (safe, 'config.json'):
            path = folder/name; path.write_bytes((component+'/'+name).encode())
            files[path.relative_to(directory).as_posix()] = path
        (folder/binary).write_bytes(b'unselected binary fixture')
    tokenizer = directory/'tokenizer'; tokenizer.mkdir()
    for name in ('vocab.json', 'merges.txt', 'tokenizer_config.json', 'special_tokens_map.json',
                 'added_tokens.json', 'tokenizer.json', 'chat_template.jinja'):
        path = tokenizer/name; path.write_bytes(('tokenizer/'+name).encode())
        files[path.relative_to(directory).as_posix()] = path
    path = directory/'scheduler'/'scheduler_config.json'; path.parent.mkdir(); path.write_bytes(b'scheduler fixture')
    files[path.relative_to(directory).as_posix()] = path
    return files


class TokenBatch(dict):
    def __getattr__(self, name):return self[name]
    def to(self, device):return TokenBatch({key: value.to(device) for key, value in self.items()})


class TinyTokenizer:
    model_max_length = 77
    pad_token_id = 0

    def __init__(self, change_padded_prefix=False):
        self.calls = []; self.change_padded_prefix = change_padded_prefix

    def __call__(self, prompt, **kwargs):
        if kwargs.get('truncation') is not False:
            raise AssertionError('Native prompt validation must explicitly disable truncation')
        self.calls.append({'prompt': prompt, **kwargs})
        values = [1]+list(range(10, 10+len(prompt.split())))+[2]
        active = len(values)
        if kwargs.get('padding') == 'max_length':
            if kwargs.get('max_length') != 77:
                raise AssertionError('The native CLIP context is exactly77 tokens')
            values += [0]*max(0, 77-len(values))
            if self.change_padded_prefix:values[1] += 1000
        return TokenBatch(input_ids=torch.tensor([values]),
            attention_mask=torch.tensor([[int(index < active) for index in range(len(values))]]))


class TinyText:
    dtype = torch.float32

    def __init__(self):self.calls = []

    def __call__(self, input_ids, **kwargs):
        self.calls.append(input_ids.detach().clone())
        value = input_ids.float()[:, :, None].expand(-1, -1, 4).contiguous()
        return SimpleNamespace(last_hidden_state=value)


class FreeScenePriorControlTests(unittest.TestCase):
    def test_local_manifest_prefers_safe_fp16_weights_and_hashes_selected_component_asset_closure(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); expected = local_fixture(directory)
            manifest = local_component_manifest(directory)
            self.assertEqual(manifest['variant'], 'fp16')
            self.assertTrue(manifest['local_files_only'])
            self.assertEqual(manifest['use_safetensors_by_component'], {'unet': True, 'vae': True, 'text_encoder': True})
            self.assertEqual(set(manifest['files']), set(expected))
            for name, path in expected.items():
                self.assertEqual(manifest['files'][name], {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                                          'bytes': path.stat().st_size})
            (directory/'unet'/'diffusion_pytorch_model.fp16.bin').write_bytes(b'changed unselected binary')
            self.assertEqual(local_component_manifest(directory), manifest)

    def test_missing_fp16_component_and_index_parent_traversal_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); local_fixture(directory)
            folder = directory/'unet'
            (folder/'diffusion_pytorch_model.fp16.safetensors').unlink()
            (folder/'diffusion_pytorch_model.fp16.bin').unlink()
            with self.assertRaises(ValueError):local_component_manifest(directory)
            # A real file outside the component directory cannot be smuggled
            # into the selected local asset closure through an index entry.
            (directory/'escape.safetensors').write_bytes(b'not a component shard')
            (folder/'diffusion_pytorch_model.fp16.safetensors.index.json').write_text(
                json.dumps({'weight_map': {'weight': '../escape.safetensors'}}), encoding='utf-8')
            with self.assertRaises(ValueError):local_component_manifest(directory)

    def test_eight_native_text_prompts_bind_single_or_four_attributes_without_reference_inputs(self):
        prompts = control_prompts()
        self.assertEqual(len(prompts), 8)
        self.assertEqual(len({row['prompt_id'] for row in prompts}), 8)
        self.assertEqual(Counter((r['category'], r['color'], r['prompt_level']) for r in prompts),
            Counter({(category, color, level): 1 for category in ('lamp', 'pillow')
                for color in ('red', 'blue') for level in ('single', 'composed')}))
        for row in prompts:
            subject = 'lampshade_color' if row['category'] == 'lamp' else 'fabric_color'
            self.assertEqual(row['requested_attributes'][subject], row['color'])
            self.assertNotRegex(row['prompt'].lower(), r'\breference\b|\bsku\b|\btarget\b|\binput image\b')
            if row['category'] == 'lamp':
                self.assertRegex(row['prompt'].lower(), r'\b'+row['color']+r'\s+lampshade\b')
            else:
                self.assertRegex(row['prompt'].lower(), r'\b'+row['color']+r'\s+(?:fabric|cushion)\b')
            if row['prompt_level'] == 'single':
                self.assertEqual(set(row['requested_attributes']), {subject})
                self.assertNotRegex(row['prompt'].lower(), r'\bwall\b|\btabletop\b|\bwooden\b|\bstone\b|\bfrom the (?:left|right)\b')
            else:
                self.assertEqual(set(row['requested_attributes']),
                    {subject, 'wall_color', 'side_light_direction', 'tabletop_material'})
                wall, direction, material = ('pale blue', 'right', 'wooden') if row['color'] == 'red' else ('warm beige', 'left', 'neutral stone')
                self.assertEqual(row['requested_attributes']['wall_color'], wall)
                self.assertEqual(row['requested_attributes']['side_light_direction'], direction)
                self.assertEqual(row['requested_attributes']['tabletop_material'], material)
                self.assertIn(wall+' wall', row['prompt'])
                self.assertIn('from the '+direction, row['prompt'])
                self.assertIn(material+' tabletop', row['prompt'])

    def test_32_records_cover_each_prompt_at_both_common_seeds_and_native_resolutions(self):
        records = control_records(); prompts = control_prompts()
        self.assertEqual(len(records), 32)
        self.assertEqual(len({row['image_id'] for row in records}), 32)
        expected = {(row['prompt_id'], resolution, seed) for row in prompts
                    for resolution in (256, 512) for seed in (1042, 1043)}
        self.assertEqual({(row['prompt_id'], row['resolution'], row['noise_seed']) for row in records}, expected)
        self.assertEqual(Counter(row['resolution'] for row in records), Counter({256: 16, 512: 16}))
        lookup = {row['prompt_id']: row for row in prompts}
        forbidden = {'reference', 'reference_path', 'input_image', 'target', 'target_image',
                     'ground_truth', 'mask', 'SKU', 'teacher_caption'}
        for row in records:
            self.assertFalse(set(row) & forbidden)
            for key in ('prompt', 'category', 'color', 'prompt_level', 'requested_attributes'):
                self.assertEqual(row[key], lookup[row['prompt_id']][key])

    def test_native_clip_checks_full_text_then_passes_exact77_token_ids_to_text_encoder(self):
        tokenizer = TinyTokenizer(); encoder = TinyText(); prompt = control_prompts()[0]['prompt']
        condition, audit = native_text_conditions(tokenizer, encoder, prompt, torch.device('cpu'))
        self.assertEqual(tuple(condition.shape), (1, 77, 4))
        self.assertTrue(torch.isfinite(condition).all())
        self.assertEqual(len(encoder.calls), 1); self.assertEqual(tuple(encoder.calls[0].shape), (1, 77))
        self.assertEqual(len(tokenizer.calls), 2)
        self.assertTrue(all(call['truncation'] is False and call['prompt'] == prompt for call in tokenizer.calls))
        expected = [1]+list(range(10, 10+len(prompt.split())))+[2]
        self.assertEqual(encoder.calls[0][0, :len(expected)].tolist(), expected)
        self.assertTrue(audit)

    def test_77_token_boundary_is_accepted_and_longer_full_prompt_is_rejected_before_encoding(self):
        tokenizer = TinyTokenizer(); encoder = TinyText()
        native_text_conditions(tokenizer, encoder, ' '.join(['word']*75), torch.device('cpu'))
        self.assertEqual(tuple(encoder.calls[0].shape), (1, 77))
        longer = TinyTokenizer(); unused_encoder = TinyText()
        with self.assertRaises(ValueError):
            native_text_conditions(longer, unused_encoder, ' '.join(['word']*76), torch.device('cpu'))
        self.assertFalse(unused_encoder.calls)
        self.assertTrue(longer.calls)
        self.assertTrue(all(call['truncation'] is False for call in longer.calls))

    def test_padded_tokenizer_prefix_cannot_change_the_original_prompt_silently(self):
        tokenizer = TinyTokenizer(change_padded_prefix=True); encoder = TinyText()
        with self.assertRaises(ValueError):
            native_text_conditions(tokenizer, encoder, control_prompts()[0]['prompt'], torch.device('cpu'))
        self.assertFalse(encoder.calls)


if __name__ == '__main__':unittest.main()
