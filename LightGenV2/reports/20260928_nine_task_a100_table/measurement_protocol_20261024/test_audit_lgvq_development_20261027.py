import hashlib,json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import audit_lgvq_development_20261027 as audit


class AuditTests(unittest.TestCase):
    def test_original_receipt_hashes_and_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'raw.txt';p.write_bytes(b'abc')
            r={'verified_files':[{'path':'raw.txt','bytes':3,'sha256':hashlib.sha256(b'abc').hexdigest()}]}
            (root/'receipt.json').write_text(json.dumps(r))
            self.assertEqual(audit.receipt(root,'receipt.json')['files'],1)
            p.write_bytes(b'abd')
            with self.assertRaises(ValueError):audit.receipt(root,'receipt.json')

    def test_score_bytes_not_tolerance(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=np.zeros((128,16),dtype=np.float32)
            np.savez(root/'outputs.npz',prediction=a)
            audit.scores(root,a)
            b=a.copy();b[-1,-1]=-0.
            np.savez(root/'outputs.npz',prediction=b)
            with self.assertRaises(ValueError):audit.scores(root,a)

    def test_auditor_does_not_import_torch(self):
        import sys
        self.assertNotIn('torch',sys.modules)


if __name__=='__main__':unittest.main()
