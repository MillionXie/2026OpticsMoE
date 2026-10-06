"""Published baseline identity metadata; no model/data or hardware execution."""
import hashlib
import json
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'LightGenV2/reports/20260915_baseline_methods'


class BaselineInventoryTests(unittest.TestCase):
    def test_manifest_identity_matches_retention_receipt(self):
        receipt = json.loads((ROOT / 'maintenance/storage/HISTORICAL_BASELINE_PAYLOAD_VISIBILITY_20261006.json').read_text(encoding='utf-8'))
        build = json.loads((BASE / 'code_packages/BUILD_MANIFEST.json').read_text(encoding='utf-8'))
        builds = {row['task']: row for row in build['tasks']}
        total = 0
        for row in receipt['packages']:
            path = BASE / 'code_packages' / row['task'] / 'SOURCE_MANIFEST.json'
            raw = path.read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), row['source_manifest_sha256'])
            manifest = json.loads(raw)
            self.assertEqual(manifest['source_commit'], row['source_commit'])
            self.assertEqual(len(manifest['files']), row['source_files'])
            self.assertEqual(builds[row['task']]['sha256'], row['zip_sha256'])
            total += len(manifest['files'])
        self.assertEqual(total, 738)

    def test_builder_document_dependencies_are_versioned(self):
        tracked = set(subprocess.check_output(['git', '-C', str(ROOT), 'ls-files', '-z']).decode().split('\0'))
        spec = json.loads((BASE / 'code_handoff.json').read_text(encoding='utf-8'))
        for task in spec['tasks']:
            self.assertIn(task['methods'], tracked)
            self.assertIn('LightGenV2/reports/20260915_baseline_methods/code_packages/' + task['directory'] + '/SOURCE_MANIFEST.json', tracked)
        self.assertIn('LightGenV2/reports/20260915_baseline_methods/VERSION_MAP.json', tracked)


if __name__ == '__main__':
    unittest.main()
