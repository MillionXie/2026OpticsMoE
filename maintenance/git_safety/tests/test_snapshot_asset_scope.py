"""Asset identity checks must neither hide reports nor change user files/refs."""
from pathlib import Path
import subprocess
import tempfile
import unittest

from maintenance.git_safety.source_snapshot_scope import audit, audit_history, blob_id, identity


class SnapshotAssetScopeTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.git('init', '-q')
        self.git('config', 'user.name', 'Fixture')
        self.git('config', 'user.email', 'fixture@example.invalid')
        self.git('config', 'core.autocrlf', 'false')
        (self.root / 'report.json').write_bytes(b'{"accuracy": 0.8}\n')
        (self.root / 'tool.py').write_bytes(b'answer = 42\n')
        self.git('add', 'report.json', 'tool.py')
        self.git('commit', '-qm', 'fixture')
        self.git('update-ref', 'refs/archive/fixture', 'HEAD')
        self.git('rm', '-q', 'report.json')
        self.git('commit', '-qm', 'historical report')
        (self.root / 'old_report.json').write_bytes(b'{"accuracy": 0.8}\n')
        (self.root / 'new_report.json').write_bytes(b'{"accuracy": 0.9}\n')
        (self.root / 'copy.py').write_bytes(b'answer = 42\r\n')

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.root), *args])

    def test_default_stays_source_only(self):
        result = audit(self.root)
        self.assertEqual([row['path'] for row in result['files']], ['copy.py'])
        self.assertEqual(result['skipped_counts']['not_selected_source_extension'], 2)

    def test_asset_history_distinguishes_unknown_report_without_mutation(self):
        before = (self.git('status', '--porcelain', '-z'), self.git('show-ref'),
                  {p.name: p.read_bytes() for p in self.root.iterdir() if p.is_file()})
        result = audit_history(self.root, include_assets=True)
        rows = {row['path']: row for row in result['files']}
        self.assertEqual(result['files_checked'], 3)
        self.assertEqual(rows['old_report.json']['history_identity'], 'exact_git_blob')
        self.assertEqual(rows['new_report.json']['history_identity'], 'no_byte_identity_in_reference')
        after = (self.git('status', '--porcelain', '-z'), self.git('show-ref'),
                 {p.name: p.read_bytes() for p in self.root.iterdir() if p.is_file()})
        self.assertEqual(before, after)
        self.assertEqual(result['mutations'], [])

    def test_large_asset_explicitly_skipped(self):
        (self.root / 'large.bin').write_bytes(b'x' * (2 * 1024 * 1024 + 1))
        result = audit(self.root, include_assets=True)
        self.assertEqual(result['skipped_counts']['file_over_2_mib_manual_review'], 1)
        self.assertNotIn('large.bin', [row['path'] for row in result['files']])

    def test_binary_identity_never_normalizes_bytes(self):
        self.assertEqual(identity(b'a\r\nb', {blob_id(b'a\nb'): ['binary']},
                                  normalize_crlf=False), ('no_byte_identity_in_reference', []))


if __name__ == '__main__':
    unittest.main()
