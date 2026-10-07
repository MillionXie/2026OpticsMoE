"""Test hooks in disposable fixture repositories, never the active workspace."""
import shutil
import subprocess
import tempfile
from pathlib import Path
import unittest

from maintenance.git_safety.install_commit_guard import install, inspect

ROOT = Path(__file__).resolve().parents[3]


class CommitGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run_git('init', '-b', 'main')
        self.run_git('config', 'user.name', 'Guard fixture')
        self.run_git('config', 'user.email', 'fixture@example.invalid')
        for name in ('.githooks/pre-commit', 'maintenance/git_safety/review_git.py'):
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        self.run_git('add', '.githooks/pre-commit', 'maintenance/git_safety/review_git.py')
        self.run_git('commit', '-m', 'fixture baseline')
        self.head = self.run_git('rev-parse', 'HEAD').stdout.strip()

    def run_git(self, *args, check=True):
        return subprocess.run(['git', '-C', str(self.root), *args], check=check,
                              capture_output=True, text=True)

    def test_hook_rejects_weight_and_permits_source(self):
        self.assertTrue(install(self.root, self.head)['installed'])
        (self.root / 'candidate.pt').write_bytes(b'fixture-weight')
        self.run_git('add', 'candidate.pt')
        result = self.run_git('commit', '-m', 'must reject asset', check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('weight/cache/transport artifact', result.stdout + result.stderr)
        self.assertEqual(self.run_git('rev-parse', 'HEAD').stdout.strip(), self.head)
        self.run_git('rm', '--cached', 'candidate.pt')
        (self.root / 'source.py').write_text('value = 1\n', encoding='utf8')
        self.run_git('add', 'source.py')
        self.assertEqual(self.run_git('commit', '-m', 'source allowed').returncode, 0)

    def test_never_overwrites_existing_hook(self):
        hook = self.root / '.git/hooks/pre-commit'
        hook.write_text('# existing user hook\n', encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'Existing pre-commit'):
            install(self.root, self.head)
        self.assertEqual(hook.read_text(encoding='utf8'), '# existing user hook\n')

    def test_hook_rejects_literal_connection_credential_without_printing_value(self):
        install(self.root, self.head)
        marker = 'fixture-not-a-real-password'
        (self.root / 'connection.py').write_text(
            "client.connect(password='" + marker + "')\n", encoding='utf8')
        self.run_git('add', 'connection.py')
        result = self.run_git('commit', '-m', 'reject fixture credential', check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('literal connection credential', result.stdout + result.stderr)
        self.assertNotIn(marker, result.stdout + result.stderr)

    def test_non_main_checkout_is_not_reconfigured(self):
        self.run_git('checkout', '--detach', self.head)
        with self.assertRaisesRegex(ValueError, 'canonical main'):
            install(self.root, self.head)

    def test_explicit_interpreter_avoids_path_lookup(self):
        import os
        state = install(self.root, self.head)
        self.assertTrue(Path(state['configured_python']).is_file())
        # Missing generic python is irrelevant: the hook uses the tested path.
        self.assertIn('lightgen.guardPython', (self.root / '.githooks/pre-commit').read_text())

    def test_foreign_interpreter_is_not_replaced(self):
        self.run_git('config', '--local', 'lightgen.guardPython', '/user/custom/python')
        with self.assertRaisesRegex(ValueError, 'Existing guard interpreter'):
            install(self.root, self.head)
        self.assertEqual(self.run_git('config', '--get', 'lightgen.guardPython').stdout.strip(), '/user/custom/python')

    def test_foreign_hook_configuration_is_preserved(self):
        self.run_git('config', 'core.hooksPath', 'user-hooks')
        with self.assertRaisesRegex(ValueError, 'Existing hooksPath'):
            inspect(self.root)
        self.assertEqual(self.run_git('config', '--get', 'core.hooksPath').stdout.strip(), 'user-hooks')

    def test_stale_head_and_occupied_index_rejected(self):
        with self.assertRaisesRegex(ValueError, 'HEAD changed'):
            install(self.root, '0' * 40)
        (self.root / 'source.py').write_text('value = 2\n', encoding='utf8')
        self.run_git('add', 'source.py')
        with self.assertRaisesRegex(ValueError, 'Index is occupied'):
            install(self.root, self.head)


if __name__ == '__main__':
    unittest.main()
