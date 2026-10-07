import hashlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[3]


class HistoricalSourceTests(unittest.TestCase):
    def test_original_sources_and_existing_optics_dependency(self):
        descriptor=json.loads((ROOT/'maintenance/storage/ADRENAL_HISTORICAL_SOURCE_ADOPTION_20261007.json').read_text())
        self.assertEqual(len(descriptor['files']),42)
        for row in descriptor['files']:
            raw=(ROOT/row['path']).read_bytes().replace(b'\r\n',b'\n')
            self.assertEqual(hashlib.sha256(raw).hexdigest(),row['source_blob_sha256'],row['path'])
            if row['path'].endswith('.py'):
                compile(raw,row['path'],'exec')
        base=ROOT/'LightGenV2/demo_check/adrenal_softsign_code_export_20260915_145336/code'
        self.assertEqual(hashlib.sha256((base/'PROTOCOL.md').read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
                         descriptor['protocol_original_sha256'])
        self.assertEqual(hashlib.sha256((base/'optical_reference/optics.py').read_bytes()).hexdigest(),
                         'b250627b14357a254eb12c785caf20f3aff2bf3fe30121a05321094803c73a7d')


if __name__=='__main__':unittest.main()
