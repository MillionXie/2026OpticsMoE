"""Small CPU-only checks of the scope adapter; no model/GPU imports."""
import ast
import inspect
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent))
import run_temporal_qwen_fullscope_20261029 as scope


class ScopeTests(unittest.TestCase):
    def test_sample_adapter_and_child_entry(self):
        suite = scope.adapt(scope.base.suite)
        self.assertIn(str(scope.ENTRY), suite.__code__.co_filename)
        self.assertIn(512, scope.adapt(scope.base.worker).__code__.co_consts)
        self.assertNotIn(200, scope.adapt(scope.base.worker).__code__.co_consts)

    def test_no_gpu_or_model_import_for_adapter(self):
        tree = ast.parse(inspect.getsource(scope))
        imports = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any('torch' in n or 'transformers' in n for n in imports))
        self.assertEqual(scope.COUNT, 512)

    def test_normalization_and_protocol_are_new(self):
        fn = scope.adapt(scope.base.summarize)
        calls = []
        for i in range(512):
            t = i * 0.125
            calls.append(dict(sample_index=i, sample_id=str(i), preprocess_start=t,
                              prepare_end=t + 0.02, model_call_start=t + 0.03,
                              first_block_start=t + 0.04, gpu_output=t + 0.08,
                              call_return=t + 0.125))
        class Sample:
            host_monotonic_s = -1.0
        last = Sample(); last.host_monotonic_s = 100.0
        # Mock energy only; exercise the unchanged serial boundary partition.
        import LightGenV2.tasks.t06_video_quality_assessment.strict_tf_window_20261020 as energy
        from unittest.mock import patch
        with patch.object(energy, 'interval_energy', return_value={'energy_j': 20000.0}):
            result = fn(calls, [Sample(), last])
        self.assertEqual(result['videos'], 512)
        self.assertEqual(result['frames'], 2048)
        self.assertAlmostEqual(result['continuous_wall_s'], 64.0)
        self.assertAlmostEqual(result['ms_per_video'], 125.0)
        self.assertAlmostEqual(result['equivalent_ms_per_input_frame'], 31.25)
        self.assertEqual(result['protocol_id'], scope.NEW_PROTOCOL)


if __name__ == '__main__':
    unittest.main()
