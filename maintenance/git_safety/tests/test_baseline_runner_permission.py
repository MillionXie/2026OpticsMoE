"""Generated handoffs cannot initialize nested repositories by default."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from LightGenV2.scripts.build_baseline_handoff import RUNNER


class RunnerPermissionTests(unittest.TestCase):
    def test_init_requires_explicit_permission_without_writing_git(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'run_baseline.py').write_text(RUNNER, encoding='utf8')
            (root / 'TASK.json').write_text(json.dumps({'actions': {}}))
            before = sorted(p.name for p in root.iterdir())
            result = subprocess.run([sys.executable, str(root / 'run_baseline.py'), 'init-source'],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('nested repository is disabled', result.stderr)
            self.assertEqual(sorted(p.name for p in root.iterdir()), before)

    def test_readonly_check_still_works_without_git_repository(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'run_baseline.py').write_text(RUNNER, encoding='utf8')
            (root / 'TASK.json').write_text(json.dumps({'actions': {}}))
            (root / 'SOURCE_MANIFEST.json').write_text(json.dumps({'files': [], 'source_commit': 'fixture'}))
            result = subprocess.run([sys.executable, str(root / 'run_baseline.py'), 'check'],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((root / 'source').exists())


if __name__ == '__main__':
    unittest.main()
