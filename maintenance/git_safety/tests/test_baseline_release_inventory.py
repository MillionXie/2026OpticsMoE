"""Published baseline identity metadata; no model/data or hardware execution."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest
import zipfile

from LightGenV2.scripts.build_baseline_handoff import RUNNER
from maintenance.storage.check_historical_baseline_assets import inspect, source_identity

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
            # Receipt pins original Windows release bytes. Git text checkout
            # may normalize line endings; recover that representation only,
            # never change per-source hashes or accept arbitrary content edits.
            release_bytes = raw.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
            self.assertEqual(hashlib.sha256(release_bytes).hexdigest(), row['source_manifest_sha256'])
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
            for name in ('README.md', 'METHODS.md', 'REFERENCE.json', 'TASK.json',
                         'requirements.txt', 'run_baseline.py'):
                self.assertIn('LightGenV2/reports/20260915_baseline_methods/code_packages/'
                              + task['directory'] + '/' + name, tracked)
        self.assertIn('LightGenV2/reports/20260915_baseline_methods/VERSION_MAP.json', tracked)

    def test_promoted_package_metadata_matches_original_spec_not_current_models(self):
        spec = json.loads((BASE / 'code_handoff.json').read_text(encoding='utf8'))
        versions = json.loads((BASE / 'VERSION_MAP.json').read_text(encoding='utf8'))
        for task in spec['tasks']:
            package = BASE / 'code_packages' / task['directory']
            self.assertEqual(json.loads((package / 'TASK.json').read_text(encoding='utf8')), task)
            reference = json.loads((package / 'REFERENCE.json').read_text(encoding='utf8'))
            selected = [dict(row) for row in versions['rows'] if row['table_order'] == task['directory'][:2]]
            for row in selected:
                row.pop('checkpoint', None)
            self.assertEqual(reference, {'model_snapshots': versions['model_snapshots'], 'rows': selected})
            # Task-page methods have evolved (notably LSP); keep the original
            # release methods, never replace them with current main text.
            with zipfile.ZipFile(BASE / 'code_packages' / (task['directory'] + '.zip')) as archive:
                original = archive.read(task['directory'] + '/METHODS.md').decode('utf8').replace('\r\n', '\n')
            self.assertEqual((package / 'METHODS.md').read_text(encoding='utf8'), original)

    def test_existing_launchers_cannot_create_nested_git_without_permission(self):
        spec = json.loads((BASE / 'code_handoff.json').read_text(encoding='utf8'))
        for task in spec['tasks']:
            package = BASE / 'code_packages' / task['directory']
            git_dir = package / 'source' / '.git'
            before = git_dir.exists()
            result = subprocess.run([sys.executable, str(package / 'run_baseline.py'), 'init-source'],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn('nested repository is disabled', result.stderr)
            self.assertEqual(git_dir.exists(), before)

    def test_existing_launchers_equal_the_guarded_template(self):
        spec = json.loads((BASE / 'code_handoff.json').read_text(encoding='utf8'))
        for task in spec['tasks']:
            self.assertEqual((BASE / 'code_packages' / task['directory'] / 'run_baseline.py')
                             .read_text(encoding='utf8').strip(), RUNNER.strip())

    def test_retained_source_and_original_zip_bytes_are_exact(self):
        report = inspect(ROOT)
        self.assertTrue(report['all_current_source_bytes_exact'])
        self.assertTrue(report['all_original_archives_verified'])
        self.assertEqual(sum(p['files'] for p in report['packages']), 738)
        self.assertFalse(report['models_or_data_evaluated'])

    def test_asset_audit_does_not_waive_eol_or_content_differences(self):
        expected = hashlib.sha256(b'config\n').hexdigest()
        self.assertEqual(source_identity(b'config\n', expected), 'exact')
        self.assertEqual(source_identity(b'config\r\n', expected), 'line_endings_only_not_accepted')
        self.assertEqual(source_identity(b'changed\n', expected), 'content_difference_not_accepted')
        self.assertEqual(source_identity(None, expected), 'missing')


if __name__ == '__main__':
    unittest.main()
