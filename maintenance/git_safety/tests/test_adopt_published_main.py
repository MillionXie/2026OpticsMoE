import subprocess
import tempfile
from pathlib import Path
import unittest

from maintenance.git_safety.adopt_published_main import adopt, audit


class AdoptionTests(unittest.TestCase):
    def fixture(self, folder):
        store = Path(folder) / 'store'
        root = Path(folder) / 'root'
        store.mkdir(); root.mkdir()
        subprocess.run(['git', '-C', str(store), 'init', '-q'], check=True)
        (store / 'entry.txt').write_text('entry\n')
        (store / 'matching.txt').write_text('same\n')
        subprocess.run(['git', '-C', str(store), 'add', 'entry.txt', 'matching.txt'], check=True)
        subprocess.run(['git', '-C', str(store), '-c', 'user.name=Fixture',
                        '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture'], check=True)
        pin = subprocess.check_output(['git', '-C', str(store), 'rev-parse', 'HEAD']).decode().strip()
        return root, store, pin

    def test_adoption_preserves_runtime_and_matching_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root, store, pin = self.fixture(folder)
            (root / 'matching.txt').write_bytes(b'same\r\n')
            (root / 'runtime').mkdir()
            (root / 'runtime' / 'best.pt').write_bytes(b'protected weight')
            result = adopt(root, store, 'git', pin)
            self.assertEqual(result['new_tracked_files'], 1)
            self.assertEqual((root / 'matching.txt').read_bytes(), b'same\r\n')
            self.assertEqual((root / 'runtime' / 'best.pt').read_bytes(), b'protected weight')
            self.assertEqual((root / 'entry.txt').read_text(), 'entry\n')

    def test_conflict_leaves_root_without_git(self):
        with tempfile.TemporaryDirectory() as folder:
            root, store, pin = self.fixture(folder)
            (root / 'entry.txt').write_text('unique user code\n')
            self.assertTrue(audit(root, store, 'git', pin)['conflicts'])
            with self.assertRaises(ValueError):
                adopt(root, store, 'git', pin)
            self.assertFalse((root / '.git').exists())
            self.assertEqual((root / 'entry.txt').read_text(), 'unique user code\n')


if __name__ == '__main__':
    unittest.main()
