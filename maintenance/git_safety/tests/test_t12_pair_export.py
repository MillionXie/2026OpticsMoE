import json
import csv
from pathlib import Path
import tempfile
import unittest

import torch
import numpy as np
from PIL import Image
from LightGenV2.tasks.t12_text_to_image.export_baseline_pairs import export_pairs, save_tensor, METADATA_KEYS


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
    def test_matches_original_csv_numpy_rounding_and_clipping(self):
        # The historical materialize_pairs.save_rgb formula; no private data needed.
        values = torch.linspace(-2, 2, 3 * 16 * 1024).reshape(3, 16, 1024).requires_grad_(True)
        pixels = (values.detach().cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5
        expected = np.clip(np.rint(pixels), 0, 255).astype(np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'image.png'
            save_tensor(values, path)
            with Image.open(path) as image:
                self.assertTrue(np.array_equal(np.asarray(image), expected))

    def test_old_csv_layout_and_prompt_column(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)/'new'
            export_pairs({'test': Dataset()}, out, limit=1, manifest_format='csv')
            with (out/'pairs.csv').open(newline='') as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row['prompt'], 'prompt')
            self.assertEqual(row['index'], '0')
            self.assertTrue((out/row['input_png']).is_file())
            self.assertFalse((out/'test').exists())

    def test_csv_rejects_all_splits_before_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)/'new'
            with self.assertRaises(ValueError):
                export_pairs({'test': Dataset(), 'val': Dataset()}, out, manifest_format='csv')
            self.assertFalse(out.exists())

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
