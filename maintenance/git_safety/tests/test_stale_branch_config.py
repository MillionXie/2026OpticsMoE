import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'retire_stale_branch_config.py'


class StaleConfigTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.git('init', '--initial-branch=main')
        self.git('config', 'user.name', 'Fixture')
        self.git('config', 'user.email', 'fixture@example.invalid')
        self.git('commit', '--allow-empty', '-m', 'fixture')
        self.git('config', 'branch.main.remote', 'origin')
        self.git('config', 'branch.old.remote', 'origin')
        self.git('config', 'branch.unknown.remote', 'origin')
        self.git('update-ref', 'refs/archive/test/old', 'HEAD')

    def tearDown(self):
        self.temp.cleanup()

    def git(self, *args):
        return subprocess.check_output(['git', *args], cwd=self.root, stderr=subprocess.DEVNULL)

    def run_tool(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=self.root,
                              capture_output=True, text=True)

    def test_plan_does_not_mutate(self):
        before = (self.root / '.git/config').read_bytes()
        result = self.run_tool()
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual([x['branch'] for x in plan['candidates']], ['old'])
        self.assertEqual(plan['unbound_preserved'], ['unknown'])
        self.assertEqual((self.root / '.git/config').read_bytes(), before)

    def test_apply_preserves_unknown_main_refs_and_backup(self):
        before = (self.root / '.git/config').read_bytes()
        refs = self.git('show-ref')
        result = self.run_tool('--apply', '--backup-dir', '.codex_tmp/fixture_backup')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['removed_sections'], 1)
        self.assertEqual(self.git('show-ref'), refs)
        self.assertEqual(self.git('config', 'branch.main.remote').strip(), b'origin')
        self.assertEqual(self.git('config', 'branch.unknown.remote').strip(), b'origin')
        self.assertEqual((self.root / '.codex_tmp/fixture_backup/config.before.private').read_bytes(), before)

    def test_backup_outside_private_rejected(self):
        before = (self.root / '.git/config').read_bytes()
        self.assertNotEqual(self.run_tool('--apply', '--backup-dir', 'public').returncode, 0)
        self.assertEqual((self.root / '.git/config').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
