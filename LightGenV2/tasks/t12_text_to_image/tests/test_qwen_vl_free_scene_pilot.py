"""CPU-only contracts for the reference-only free-scene generation pilot."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from ..qwen_vl_free_scene_pilot import (audit_tokens, generation_inputs,
    load_reference_inputs, prompt_conditions, reference_seeds, select_references)


def synthetic_metadata():
    rows = []
    for split, count in (('train', 960), ('val', 96)):
        for category in ('lamp', 'pillow'):
            for index in range(count//2):
                product = (index//6) % 4
                rows.append({'split': split, 'category': category,
                    'render_protocol': 'controlled_room_edit_v1',
                    'reference_context_id': f'{split}-{category}-{product}-context{index//3 % 2}',
                    'source': {'product_id': f'{split}-{category}-SKU{product}'},
                    'target': {'product_id': 'TARGET_MUST_NOT_ENTER_MODEL', 'image_path': '/nonexistent/target.png'},
                    'teacher_caption': 'PRIVATE TARGET CAPTION', 'edit_mask_path': '/nonexistent/edit_mask.png'})
    return rows


def prepared_fixture(directory):
    rows = synthetic_metadata()
    pair_path = directory/'annotated_pairs.json'
    pair_path.write_text(json.dumps(rows), encoding='utf-8')
    selected = select_references(rows); manifest = {}; values = []
    for index, record in enumerate(selected):
        value = np.empty((256, 256, 3), dtype=np.uint8)
        value[:] = (20+index*7, 40+index*3, 120-index)
        relative = f"images/{record['split']}/reference/{record['metadata_row_index']:05d}.png"
        path = directory/relative; path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(value).save(path)
        manifest[relative] = hashlib.sha256(path.read_bytes()).hexdigest(); values.append(value)
    (directory/'image_sha256.json').write_text(json.dumps(manifest), encoding='utf-8')
    (directory/'report.json').write_text(json.dumps({'complete': True, 'stage': 'CPU_data_preparation_only',
        'pair_sha256': hashlib.sha256(pair_path.read_bytes()).hexdigest()}), encoding='utf-8')
    (directory/'data_audit.json').write_text(json.dumps({'protocol': 'controlled_room_edit_v1',
        'expected_counts': {'train': 960, 'val': 96}, 'role_disjoint': True, 'test_opened': False}), encoding='utf-8')
    return selected, values, manifest


class TinyTokenizer:
    def __init__(self):self.words = {}

    def convert_tokens_to_ids(self, token):
        return {'<|vision_end|>': 8001, '<|im_end|>': 8002}[token]

    def ids(self, value, truncate=False):
        item = json.loads(value)
        words = [self.words.setdefault(word, 10+len(self.words)) for word in item['prompt'].split()]
        if truncate:words = words[:12]
        image = [8000]*item['image'].count('<image>')
        if item['first'] == 'image':return [8003]+image+[8001]+words+[8002, 8004]
        return [8003]+words+image+[8001, 8002, 8004]

    def __call__(self, value, return_tensors=None, truncation=None):
        if truncation is not False:raise AssertionError('The full-prompt tokenizer check must disable truncation')
        return {'input_ids': torch.tensor([self.ids(value)])}


class TinyProcessor:
    image_token = '<image>'

    def __init__(self, truncate_actual=False):
        self.tokenizer = TinyTokenizer(); self.truncate_actual = truncate_actual

    def apply_chat_template(self, messages, **kwargs):
        content = messages[0]['content']
        return json.dumps({'first': content[0]['type'], 'image': self.image_token,
            'prompt': next(row['text'] for row in content if row['type'] == 'text')})

    def __call__(self, text, images, return_tensors=None):
        expanded = text[0].replace(self.image_token, self.image_token*64)
        ids = torch.tensor([self.tokenizer.ids(expanded, self.truncate_actual)])
        return {'input_ids': ids, 'attention_mask': torch.ones_like(ids),
            'pixel_values': torch.zeros(1, 3), 'image_grid_thw': torch.tensor([[1, 16, 16]])}


class FreeScenePilotTests(unittest.TestCase):
    def test_selection_uses_first_two_distinct_products_per_category_and_original_split(self):
        rows = synthetic_metadata(); selected = select_references(rows)
        self.assertEqual(selected, select_references(rows))
        self.assertEqual(len(selected), 8)
        self.assertEqual(Counter((r['split'], r['category']) for r in selected),
            Counter({(s, c): 2 for s in ('train', 'val') for c in ('lamp', 'pillow')}))
        self.assertEqual([r['metadata_row_index'] for r in selected], [0, 6, 480, 486, 960, 966, 1008, 1014])
        self.assertEqual(len({r['reference_context_id'] for r in selected}), 8)
        for record in selected:
            self.assertEqual(set(record), {'reference_id', 'metadata_row_index', 'split', 'category', 'reference_context_id'})

    def test_selection_rejects_test_rows_changed_protocol_and_duplicate_reference_contexts(self):
        rows = synthetic_metadata()
        test_rows = [dict(row) for row in rows]; test_rows[0]['split'] = 'test'
        with self.assertRaises(ValueError):select_references(test_rows)
        other_protocol = [dict(row) for row in rows]; other_protocol[0]['render_protocol'] = 'unreviewed'
        with self.assertRaises(ValueError):select_references(other_protocol)
        duplicate = [dict(row) for row in rows]
        duplicate[6]['reference_context_id'] = duplicate[0]['reference_context_id']
        with self.assertRaises(ValueError):select_references(duplicate)

    def test_category_generic_prompts_keep_four_bound_attributes_and_full_shape_anchor(self):
        for category, color_key in (('lamp', 'lampshade_color'), ('pillow', 'fabric_color')):
            conditions = prompt_conditions(category)
            self.assertEqual(len(conditions), 2)
            self.assertEqual(len({r['condition_id'] for r in conditions}), 2)
            for record, values in zip(conditions, (('red', 'pale blue', 'right', 'wooden'),
                                                  ('blue', 'warm beige', 'left', 'neutral stone'))):
                attrs = record['requested_attributes']
                self.assertEqual(set(attrs), {color_key, 'wall_color', 'side_light_direction', 'tabletop_material'})
                self.assertEqual(tuple(attrs[k] for k in (color_key, 'wall_color', 'side_light_direction', 'tabletop_material')), values)
                self.assertTrue(record['prompt'].startswith('Generate a new product scene'))
                self.assertIn('Retain its overall silhouette', record['prompt'])
                self.assertIn('overall silhouette', record['shape_anchor'])
                self.assertGreater(len(record['prompt'].split()), 30)
                self.assertTrue(record['prompt'].endswith('background objects may vary naturally.'))
                for value in values:self.assertIn(value, record['prompt'])
            self.assertEqual(prompt_conditions(category), conditions)
        with self.assertRaises(ValueError):prompt_conditions('table')

    def test_seed_zero_is_common_while_other_seeds_use_only_exact_reference_rgb(self):
        reference = np.zeros((256, 256, 3), dtype=np.uint8)
        tensor = torch.from_numpy(reference).permute(2, 0, 1).contiguous()
        original = reference_seeds(reference)
        self.assertEqual(original, reference_seeds(tensor))
        self.assertEqual(original, reference_seeds(Image.fromarray(reference)))
        self.assertEqual(len(set(original)), 4); self.assertEqual(original[0], 1042)
        changed = reference.copy(); changed[0, 0, 0] = 1
        other = reference_seeds(changed)
        self.assertEqual(other[0], 1042)
        self.assertTrue(all(a != b for a, b in zip(original[1:], other[1:])))
        self.assertEqual(reference_seeds(reference.copy()), original)
        for invalid in (reference.astype(np.float32), tensor.float(), np.zeros((255, 256, 3), dtype=np.uint8)):
            with self.assertRaises(ValueError):reference_seeds(invalid)
        with self.assertRaises(ValueError):reference_seeds(reference, count=3)

    def test_loader_reads_only_eight_reference_pngs_and_verifies_native_pixels_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); selected, expected, manifest = prepared_fixture(directory)
            real_open = Image.open; opened = []
            def record_open(path, *args, **kwargs):
                opened.append(Path(path))
                return real_open(path, *args, **kwargs)
            with patch('LightGenV2.tasks.t12_text_to_image.qwen_vl_free_scene_pilot.Image.open', side_effect=record_open):
                references, tensors, provenance = load_reference_inputs(directory)
            self.assertEqual(len(opened), 8)
            self.assertTrue(all(path.parent.name == 'reference' for path in opened))
            self.assertEqual(provenance['reference_PNGs_read'], 8)
            self.assertEqual(provenance['target_or_mask_PNGs_read'], 0)
            self.assertFalse(provenance['independent_TEST540_opened'])
            for record, tensor, pixels in zip(references, tensors, expected):
                self.assertEqual(tuple(tensor.shape), (3, 256, 256)); self.assertEqual(tensor.dtype, torch.uint8)
                torch.testing.assert_close(tensor, torch.from_numpy(pixels).permute(2, 0, 1), rtol=0, atol=0)
                self.assertEqual(record['reference_RGB_sha256'], hashlib.sha256(pixels.tobytes()).hexdigest())
                self.assertEqual(record['prepared_reference_PNG_sha256'], manifest[record['prepared_reference_path']])
                self.assertFalse(set(record) & {'source', 'target', 'teacher_caption', 'edit_mask_path', 'SKU'})
            Image.new('RGB', (256, 256), (255, 0, 0)).save(opened[0])
            with self.assertRaises(ValueError):load_reference_inputs(directory)

    def test_actual_model_input_helper_whitelists_full_reference_prompt_for_all64_records(self):
        selected = select_references(synthetic_metadata()); records = []
        for index, reference in enumerate(selected):
            image = torch.full((3, 256, 256), index, dtype=torch.uint8)
            schedules = []
            for condition in prompt_conditions(reference['category']):
                seeds = reference_seeds(image); schedules.append(seeds)
                for seed in seeds:
                    inputs = generation_inputs(image, condition['prompt'])
                    self.assertEqual(set(inputs), {'reference', 'rows'})
                    self.assertEqual(inputs['rows'], [{'prompt': condition['prompt']}])
                    self.assertEqual(tuple(inputs['reference'].shape), (1, 3, 256, 256))
                    torch.testing.assert_close(inputs['reference'][0], image, rtol=0, atol=0)
                    records.append((reference['reference_id'], condition['condition_id'], seed, condition['prompt']))
            self.assertEqual(schedules[0], schedules[1])
        self.assertEqual(len(records), 64)
        self.assertEqual(len({(r[0], r[1], r[2]) for r in records}), 64)
        for category in ('lamp', 'pillow'):
            prompts = [prompt_conditions(r['category']) for r in selected if r['category'] == category]
            self.assertTrue(all(value == prompts[0] for value in prompts))
        image = torch.zeros((3, 256, 256), dtype=torch.uint8); prompt = prompt_conditions('lamp')[0]['prompt']
        with self.assertRaises(ValueError):generation_inputs({'reference': image, 'target': image}, prompt)
        with self.assertRaises(TypeError):generation_inputs(image, prompt, target=image)
        with self.assertRaises(ValueError):generation_inputs(image, {'prompt': prompt, 'teacher_caption': 'private'})
        with self.assertRaises(ValueError):generation_inputs(image.float(), prompt)

    def test_actual_token_audit_retains_long_generate_prompt_and_detects_12word_truncation(self):
        prompt = prompt_conditions('lamp')[0]['prompt']
        image = Image.new('RGB', (256, 256), 'gray')
        layout = {'image_token_id': 8000, 'positions': list(range(1, 65)), 'instruction_bounds': [66, 2]}
        result = audit_tokens(TinyProcessor(), image, prompt, layout)
        self.assertTrue(result['complete_untruncated_tokenizer_identity'])
        self.assertEqual(result['prompt_word_count'], len(prompt.split()))
        self.assertGreater(result['instruction_token_count'], 30)
        with self.assertRaises(ValueError):audit_tokens(TinyProcessor(truncate_actual=True), image, prompt, layout)


if __name__ == '__main__':unittest.main()
