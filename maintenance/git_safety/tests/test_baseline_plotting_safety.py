"""Synthetic local fixtures; never execute the packager on historical reports."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT / 'LightGenV2/scripts/build_baseline_plotting_bundle.py'
spec = importlib.util.spec_from_file_location('baseline_plotting', PATH)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class BaselineSafetyTests(unittest.TestCase):
    def test_existing_delivery_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'delivery'
            output.mkdir()
            evidence = output / 'timing.csv'
            evidence.write_bytes(b'original timing\n')
            with self.assertRaises(FileExistsError):
                tool.prepare_output(output, [], [])
            self.assertEqual(evidence.read_bytes(), b'original timing\n')

    def test_missing_inputs_do_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'new'
            with self.assertRaises(FileNotFoundError):
                tool.prepare_output(output, [Path(directory) / 'missing.json'], [])
            self.assertFalse(output.exists())

    def test_input_tree_cannot_be_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                tool.prepare_output(root / 'new', [], [root])
            self.assertFalse((root / 'new').exists())

    def test_fresh_output_and_second_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.json'
            source.write_bytes(b'{}')
            output = root / 'new'
            self.assertEqual(tool.prepare_output(output, [source], []), output.resolve())
            with self.assertRaises(FileExistsError):
                tool.prepare_output(output, [source], [])
            self.assertEqual(source.read_bytes(), b'{}')

    def test_cli_requires_explicit_new_output(self):
        with self.assertRaises(SystemExit) as error:
            tool.main([])
        self.assertEqual(error.exception.code, 2)

    def test_historical_build_rules_unchanged(self):
        previous = subprocess.check_output(['git', '-C', str(ROOT), 'show',
            '8bef68b0adf880f05ad2bdda0ef02a52db57285e:LightGenV2/scripts/build_baseline_plotting_bundle.py']).decode('utf8')
        def functions(text):
            return {node.name: ast.dump(node, include_attributes=False)
                    for node in ast.parse(text).body if isinstance(node, ast.FunctionDef) and node.name != 'main'}
        before, after = functions(previous), functions(PATH.read_text(encoding='utf8'))
        self.assertTrue(set(before) <= set(after))
        for name, body in before.items():
            self.assertEqual(after[name], body, name)
        self.assertFalse(any(isinstance(node, ast.Attribute) and node.attr == 'rmtree'
                             for node in ast.walk(ast.parse(PATH.read_text(encoding='utf8')))))

    def test_untimed_retrieval_row_stays_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, out = root / 'raw', root / 'output'
            source = raw / 'abo_image_to_text'
            source.mkdir(parents=True)
            out.mkdir()
            (source / 'baseline_report.json').write_text(json.dumps({'performance': {'recall_at_1': .5}}))
            (source / 'retrieval_predictions.csv').write_text('sample_id,true_rank\na,1\nb,2\n')
            (source / 'timing_per_sample.csv').write_text('sample_id,cuda_ms,host_ms\na,3,4\n')
            (source / 'per_product_metrics.json').write_text('{}')
            (source / 'baseline_overview.png').write_bytes(b'synthetic fixture, not an image')
            with patch.object(tool, 'RAW', raw), patch.object(tool, 'OUT', out):
                summary = tool.build_abo_image_to_text()
            rows = tool.read_csv(out / 'abo_image_to_text/data.csv')
            self.assertEqual(summary['primary_value'], .5)
            self.assertEqual(rows[0]['cuda_ms'], '3')
            self.assertEqual(rows[1]['cuda_ms'], '')
            self.assertEqual(rows[1]['timing_measured'], 'False')
            self.assertEqual(rows[1]['reciprocal_rank'], '0.5')


if __name__ == '__main__':
    unittest.main()
