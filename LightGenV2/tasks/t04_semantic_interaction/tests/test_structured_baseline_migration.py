"""CPU protocol guards without importing Torch, Qwen, data or checkpoints."""
import argparse
import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'baseline_structured_5090d.py'


def functions():
    tree = ast.parse(SOURCE.read_text(encoding='utf8'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {'main', 'train_head', 'evaluate_and_time'}]
    for node in nodes:
        node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    ns = {'Path': Path, 'argparse': argparse}
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), 'exec'), ns)
    return ns


class StructuredBaselineMigrationTests(unittest.TestCase):
    def setUp(self):
        self.ns = functions()

    def test_default_protocol_and_gpu_are_unchanged(self):
        values = []
        self.ns['train_head'] = values.append
        with patch('sys.argv', ['baseline', 'train', '--model', 'm', '--data-root', 'd', '--cache-dir', 'c', '--run-dir', 'r']):
            self.assertEqual(self.ns['main'](), 0)
        self.assertEqual((values[0].baseline_protocol, values[0].expected_gpu, values[0].warmup_forwards), ('legacy', 'NVIDIA GeForce RTX 5090 D', 50))

    def test_aligned_protocol_is_explicit_opt_in(self):
        values = []
        self.ns['train_head'] = values.append
        with patch('sys.argv', ['baseline', 'train', '--model', 'm', '--data-root', 'd', '--cache-dir', 'c', '--run-dir', 'r', '--baseline-protocol', 'aligned_label_only']):
            self.ns['main']()
        self.assertEqual(values[0].baseline_protocol, 'aligned_label_only')

    def test_unknown_training_protocol_rejected_before_cache_load(self):
        with self.assertRaises(ValueError):
            self.ns['train_head'](argparse.Namespace(baseline_protocol='mistyped'))

    def test_training_preserves_existing_best_last_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('best_checkpoint.pt', 'last_checkpoint.pt', 'training_report.json'):
                path = root / name
                path.touch()
                with self.subTest(name=name), self.assertRaises(FileExistsError):
                    self.ns['train_head'](argparse.Namespace(run_dir=root, cache_dir=root))
                path.unlink()

    def test_measurement_preserves_all_old_outputs_before_gpu_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('baseline_report.json', 'timing_samples.csv', 'power_samples.csv'):
                path = root / name
                path.touch()
                with self.subTest(name=name), self.assertRaises(FileExistsError):
                    self.ns['evaluate_and_time'](argparse.Namespace(run_dir=root))
                path.unlink()

    def test_invalid_measurement_arguments_rejected_before_gpu_access(self):
        with tempfile.TemporaryDirectory() as directory:
            for gpu, warmup, samples in [(' ', 50, 200), ('5090', -1, 200), ('5090', 50, 0)]:
                with self.subTest(gpu=gpu, warmup=warmup, samples=samples), self.assertRaises(ValueError):
                    self.ns['evaluate_and_time'](argparse.Namespace(run_dir=Path(directory), expected_gpu=gpu, warmup_forwards=warmup, timing_samples=samples))

    def test_checkpoint_identity_and_actual_power_are_explicit(self):
        text = SOURCE.read_text(encoding='utf8')
        for value in ['checkpoint.get("baseline_protocol", "legacy")', 'Unknown checkpoint baseline protocol',
                      'NvidiaSmiPowerSampler(gpu_index=physical_gpu)', 'power_limit_w=actual_power_limit',
                      'head.load_state_dict(checkpoint["head"], strict=True)', 'head_type.__name__']:
            self.assertIn(value, text)

    def test_preparation_stays_outside_timed_forward(self):
        tree = ast.parse(SOURCE.read_text(encoding='utf8'))
        evaluate = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'evaluate_and_time')
        forward = next(n for n in evaluate.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
        self.assertNotIn('_prepare_one', ast.unparse(forward))
        text = ast.unparse(evaluate)
        self.assertIn('inputs = prepare(rows[index % len(rows)])', text)


if __name__ == '__main__':
    unittest.main()
