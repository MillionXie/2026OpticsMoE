"""Pure CPU guards for immutable result preservation and replay."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('delivery_audit',HERE/'audit_rental_energy_LightGen_20261029.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

class DeliveryTests(unittest.TestCase):
    def test_no_model_import(self):self.assertNotIn('torch',sys.modules)
    def test_counts(self):
        self.assertEqual(len(set(audit.EXPECTED)),12)
        self.assertEqual(audit.EXPECTED[-1],'deepseek_cap100_repeat03')
    def test_close_rejects_real_difference(self):
        audit.close(1.,1.+1e-9)
        with self.assertRaises(ValueError):audit.close(1.,1.01)
    def test_exact_preserved_bytes_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'raw.csv').write_bytes(b'actual,1\n')
            r=dict(status='complete_original_root_local_and_lab_SHA_verified',files=[dict(name='raw.csv',bytes=9,sha256=hashlib.sha256(b'actual,1\n').hexdigest())])
            (root/'whole_root_backup_receipt_complete01.json').write_text(json.dumps(r))
            self.assertEqual(len(audit.preserved(root)),64)
            (root/'raw.csv').write_bytes(b'actual,2\n')
            with self.assertRaises(ValueError):audit.preserved(root)
    def test_unsafe_member(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);r=dict(status='complete_original_root_local_and_lab_SHA_verified',files=[dict(name='../foreign',bytes=0,sha256='0'*64)])
            (root/'whole_root_backup_receipt_complete01.json').write_text(json.dumps(r))
            with self.assertRaises(ValueError):audit.preserved(root)
    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'receipt.json';audit.save(p,dict(real=True))
            with self.assertRaises(FileExistsError):audit.save(p,dict(real=False))
    def test_independent_overlap_integration_accepts_real_long_gap(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'telemetry.csv').write_text('host_monotonic_s,watts\n0,100\n0.4,200\n0.6,300\n')
            audit.independent_held_board(root,dict(start_monotonic_s=.1,end_monotonic_s=.5,gpu_energy_j=50.,estimated_system_energy_j=50.+.4*338.2))
            with self.assertRaises(ValueError):audit.independent_held_board(root,dict(start_monotonic_s=.1,end_monotonic_s=.5,gpu_energy_j=40.,estimated_system_energy_j=50.+.4*338.2))

if __name__=='__main__':unittest.main()
