"""Data/AST checks only; never instantiate a model or CUDA context."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import lgvq_async_worker_20261026 as worker


class WorkerIntegrationTests(unittest.TestCase):
    def test_complete_2048_score_identity_and_cycles(self):
        array = np.tile(np.arange(512, dtype=np.float32).reshape(32, 16), (4, 1))
        worker.validate_predictions(array, array.copy())
        altered = array.copy(); altered[-1, -1] += 1
        with self.assertRaises(ValueError):worker.validate_predictions(altered, array)
        bad_cycles = np.arange(2048, dtype=np.float32).reshape(128, 16)
        with self.assertRaisesRegex(ValueError, 'cycles'):worker.validate_predictions(bad_cycles, bad_cycles.copy())

    def test_bitwise_signed_zero_difference_is_rejected(self):
        reference = np.zeros((128, 16), dtype=np.float32)
        changed = reference.copy(); changed[0, 0] = -0.
        self.assertTrue(np.array_equal(reference, changed))
        with self.assertRaisesRegex(ValueError, 'bitwise'):worker.validate_predictions(changed, reference)
        changed=reference.copy();changed[32,0]=-0.
        with self.assertRaisesRegex(ValueError,'cycles'):worker.validate_predictions(changed,changed.copy())

    def test_probe_or_wrong_dtype_cannot_qualify(self):
        with self.assertRaises(ValueError):worker.worker(None, None, 224)
        with self.assertRaises(ValueError):worker.validate_predictions(np.zeros((32, 16)), np.zeros((32, 16)))
        with self.assertRaises(ValueError):worker.validate_predictions(np.zeros((128, 16)), np.zeros((128, 16)))

    def test_no_gpu_import_or_cli_dispatch_on_import(self):
        tree = ast.parse(Path(worker.__file__).read_text(encoding='utf-8'))
        imports = [a.name for n in tree.body if isinstance(n, ast.Import) for a in n.names]
        self.assertEqual(imports, ['hashlib', 'json', 'time'])
        self.assertFalse(any(isinstance(n, ast.If) for n in tree.body))

    def test_raw_events_preserve_both_cuda_clocks_and_host_fields(self):
        origin = SimpleNamespace(elapsed_time=lambda event: event.value)
        event = lambda v:SimpleNamespace(value=v)
        row = dict(field=7, mode='test', host_submission_end=100.,
            events=[dict(kind='frontend', label='prefix', start=event(2.), end=event(3.), stream='frontend')],
            original_H2D_events=(event(.5), event(1.)), score_ready=event(5.))
        saved = worker.serialize_field(origin, row)
        self.assertEqual(saved['original_H2D_cuda_ms'], [.5, 1.])
        self.assertEqual(saved['events'][0]['cuda_start_ms'], 2.)
        self.assertEqual(saved['score_ready_cuda_ms'], 5.)
        self.assertEqual(saved['host_submission_end'], 100.)


if __name__ == '__main__':
    unittest.main()
