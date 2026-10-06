import json
from pathlib import Path
import tempfile
import unittest

import torch
from PIL import Image
from LightGenV2.tasks.t12_text_to_image.export_baseline_pairs import export_pairs, METADATA_KEYS


class Dataset:
    def __init__(self, count=2304):
        self.count = count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return {**{key: key for key in METADATA_KEYS},
                'reference': torch.full((3, 2, 2), -1.),
                'target': torch.ones(3, 2, 2)}


class PairExportTests(unittest.TestCase):
    def test_export_preserves_roles_quantization_and_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'new'
            self.assertEqual(export_pairs({'test': Dataset()}, out, limit=1), {'test': 1})
            row = json.loads((out / 'test/pairs.jsonl').read_text())
            self.assertEqual(row['pair_id'], 'test_00000')
            self.assertEqual(row['prompt'], 'prompt')
            self.assertEqual(Image.open(out / 'test' / row['input_path']).getpixel((0, 0)), (0, 0, 0))
            self.assertEqual(Image.open(out / 'test' / row['target_path']).getpixel((0, 0)), (255, 255, 255))
            with self.assertRaises(FileExistsError):
                export_pairs({'test': Dataset()}, out, limit=1)

    def test_invalid_count_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'new'
            with self.assertRaises(ValueError):
                export_pairs({'test': Dataset(1)}, out)
            self.assertFalse(out.exists())

    def test_negative_limit_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'new'
            with self.assertRaises(ValueError):
                export_pairs({'test': Dataset()}, out, limit=-1)
            self.assertFalse(out.exists())
