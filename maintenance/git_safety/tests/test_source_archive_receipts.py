import hashlib
import tempfile
import unittest
from pathlib import Path
from maintenance.git_safety.check_source_archives import confined, inspect_manifest


class ArchiveTests(unittest.TestCase):
    def fixture(self, root):
        (root / 'old').mkdir()
        (root / 'archive').mkdir()
        (root / 'archive/source.py').write_bytes(b'original source\n')
        (root / 'old/timing.json').write_bytes(b'original timing\n')
        return {'archive_directory': 'archive', 'original_directory': 'old',
                'files': [{'name': 'source.py', 'sha256': hashlib.sha256(b'original source\n').hexdigest()}],
                'preserved_original_reports': {'old/timing.json': hashlib.sha256(b'original timing\n').hexdigest()}}

    def test_exact_archive_and_report_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertEqual(inspect_manifest(root, self.fixture(root))['errors'], [])

    def test_modified_archive_and_report_are_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); manifest = self.fixture(root)
            (root / 'archive/source.py').write_bytes(b'changed')
            (root / 'old/timing.json').write_bytes(b'changed')
            result = inspect_manifest(root, manifest)
            self.assertEqual(len(result['errors']), 2)

    def test_reoccupied_original_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); manifest = self.fixture(root)
            (root / 'old/source.py').write_bytes(b'user edit')
            self.assertEqual(len(inspect_manifest(root, manifest)['errors']), 1)
            self.assertEqual((root / 'old/source.py').read_bytes(), b'user edit')

    def test_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                confined(Path(folder), '../outside')


if __name__ == '__main__':
    unittest.main()
