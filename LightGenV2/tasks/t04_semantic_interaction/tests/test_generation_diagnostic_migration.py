"""CPU-only historical diagnostic contracts; does not import Torch or a model."""
import argparse
import ast
import json
from pathlib import Path
import re
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / 'baseline_5090d.py'


def functions():
    tree = ast.parse(SOURCE.read_text(encoding='utf8'))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {'_prompt', '_parse_array', '_parse', '_sample_metrics', 'main', 'run'}:
            node.decorator_list = []
            selected.append(node)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *selected], type_ignores=[])
    namespace = {'np': np, 're': re, 'json': json, 'argparse': argparse, 'Path': Path,
                 'ICON_SPECS': [SimpleNamespace(index=i, name=f'icon{i}') for i in range(1, 17)]}
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), 'exec'), namespace)
    return namespace


class GenerationDiagnosticMigrationTests(unittest.TestCase):
    def setUp(self):
        self.ns = functions()

    def test_default_prompt_remains_full_grid(self):
        self.assertEqual(self.ns['_prompt']('move'), self.ns['_prompt']('move', 'full_grid'))
        self.assertIn('exactly two row-major arrays of 36 integers', self.ns['_prompt']('move'))

    def test_full_grid_parser_and_original_metric(self):
        target = np.arange(36) % 17
        edit = np.arange(36) % 2
        pred, mask = self.ns['_parse'](json.dumps({'target': target.tolist(), 'edit': edit.tolist()}))
        np.testing.assert_array_equal(pred.flatten(), target)
        np.testing.assert_array_equal(mask.flatten(), edit)
        metrics = self.ns['_sample_metrics'](pred, mask, pred, mask)
        self.assertEqual(metrics['scene_exact_match'], 1.0)

    def test_sparse_opt_in_does_not_mutate_input(self):
        source = np.ones((6, 6), dtype=np.int64)
        pred, edit = self.ns['_parse']('{"changes":[[0,1,0],[5,5,16]]}', source_grid=source, output_contract='sparse_changes')
        self.assertEqual(pred[0, 1], 0)
        self.assertEqual(pred[5, 5], 16)
        self.assertEqual(edit.sum(), 2)
        np.testing.assert_array_equal(source, np.ones((6, 6)))

    def test_sparse_invalid_outputs_rejected(self):
        for text in ['{"changes":[[0,0,1],[0,0,2]]}', '{"changes":[[6,0,1]]}', '{"changes":[[0,0,17]]}', '{"changes":[[0,0,1],[1,1,2],[2,2,3]]}']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.ns['_parse'](text, source_grid=np.zeros((6, 6)), output_contract='sparse_changes')
        with self.assertRaises(ValueError):
            self.ns['_parse']('{"changes":[]}', output_contract='sparse_changes')

    def test_cli_keeps_legacy_gpu_and_protocol_defaults(self):
        captured = []
        self.ns['run'] = lambda args: captured.append(args)
        with patch('sys.argv', ['diagnostic', '--model', 'm', '--data-root', 'd', '--run-dir', 'r']):
            self.assertEqual(self.ns['main'](), 0)
        args = captured[0]
        self.assertEqual((args.expected_gpu, args.output_contract, args.warmup_forwards, args.image_size, args.max_new_tokens), ('5090', 'full_grid', 50, 224, 192))
        self.assertIsNone(args.max_samples)

    def test_output_protection_precedes_any_gpu_or_model_action(self):
        with tempfile.TemporaryDirectory() as existing:
            args = argparse.Namespace(run_dir=Path(existing), warmup_forwards=50)
            with self.assertRaises(FileExistsError):
                self.ns['run'](args)

    def test_invalid_measurement_options_fail_before_gpu(self):
        with tempfile.TemporaryDirectory() as root:
            for kwargs in [{'expected_gpu': ''}, {'warmup_forwards': -1}, {'max_samples': 0}]:
                args = argparse.Namespace(run_dir=Path(root) / 'unused', warmup_forwards=50)
                for key, value in kwargs.items():
                    setattr(args, key, value)
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    self.ns['run'](args)

    def test_real_gpu_power_mapping_is_explicit_and_diagnostic_label_present(self):
        source = SOURCE.read_text(encoding='utf8')
        self.assertIn('NvidiaSmiPowerSampler(gpu_index=physical_gpu)', source)
        self.assertIn('power_limit_w=actual_power_limit', source)
        self.assertIn('zero_shot_generation_diagnostic_not_structured_baseline', source)


if __name__ == '__main__':
    unittest.main()
