"""CPU-only protection of the historical latency/power join entry."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('power_join', ROOT / 'LightGenV2/scripts/consolidate_narrow_optical_power.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PowerJoinTests(unittest.TestCase):
    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'retained'
            out.mkdir()
            sentinel = out / 'report.json'
            sentinel.write_bytes(b'original measurement')
            args = ['join', '--latency-dir', tmp, '--power-dir', tmp, '--output-dir', str(out)]
            with patch.object(sys, 'argv', args), self.assertRaises(FileExistsError):
                module.main()
            self.assertEqual(sentinel.read_bytes(), b'original measurement')

    def test_categories_keep_parallel_residual_separate(self):
        self.assertEqual(module.category('vision_parallel_residual_nn'), 'parallel_residual')
        self.assertEqual(module.category('bridge_nn_only'), 'bridge')
        self.assertEqual(module.category('task_head_nn'), 'task_head')
        with self.assertRaises(KeyError):
            module.category('unrecognized_kernel')


if __name__ == '__main__':
    unittest.main()
