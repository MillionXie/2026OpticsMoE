import ast
from pathlib import Path
import unittest
from qwen_shared_serial_20261026 import validate_shared_snapshot


class SharedSerial(unittest.TestCase):
    def test_other_application_observed_not_rejected(self):
        row=dict(clients=[123],foreign_gpu_processes=[['456','GPU','12000']])
        self.assertEqual(validate_shared_snapshot(row,123),row['foreign_gpu_processes'])
    def test_private_MPS_client_still_rejected(self):
        with self.assertRaises(RuntimeError):validate_shared_snapshot(dict(clients=[123,456],foreign_gpu_processes=[]),123)
    def test_native_strict_worker_reused(self):
        source=Path(__file__).with_name('qwen_shared_serial_20261026.py').read_text(encoding='utf-8');ast.parse(source)
        self.assertIn('strict.worker(args)',source)
        self.assertNotIn('prepare_batch(',source)
        self.assertNotIn('while idle<12',source)
    def test_own_daemon_cleanup_does_not_wait_other_GPU_users(self):
        source=Path(__file__).with_name('qwen_shared_serial_20261026.py').read_text(encoding='utf-8')
        self.assertNotIn('engine.daemon(False)',source)
        self.assertIn('not_uncontended_baseline_performance=True',source)


if __name__=='__main__':unittest.main()
