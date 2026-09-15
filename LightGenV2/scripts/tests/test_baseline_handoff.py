"""Source export integrity and dependency collection, without GPU dependencies."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from LightGenV2.scripts.build_baseline_handoff import Snapshot, RUNNER


def snapshot(files):
    result = Snapshot.__new__(Snapshot)
    result.commit = 'fixture'
    result.files = {path: 'fixture-blob' for path in files}
    result.data = {path: text.encode('utf-8') for path, text in files.items()}
    return result


class ExportTests(unittest.TestCase):
    def test_relative_imports_package_initializers_and_dynamic_module(self):
        files = {
            'LightGenV2/__init__.py': '',
            'LightGenV2/tasks/demo/__init__.py': 'from . import registration\n',
            'LightGenV2/tasks/demo/main.py': 'from .helper import value\nimport importlib\nimportlib.import_module("experiments.shared.tool")\n',
            'LightGenV2/tasks/demo/helper.py': 'value=1\n',
            'LightGenV2/tasks/demo/registration.py': '',
            'experiments/shared/tool.py': 'from . import support\n',
            'experiments/shared/support.py': '',
            'LightGenV2/tasks/demo/runs/private.py': 'raise RuntimeError("must not export")\n',
        }
        selected = snapshot(files).closure(['LightGenV2/tasks/demo/main.py'])
        self.assertEqual(set(selected), set(files) - {'LightGenV2/tasks/demo/runs/private.py'})

    def test_configuration_chain_crosses_packages(self):
        files = {
            'LightGenV2/tasks/demo/main.py': '',
            'LightGenV2/tasks/demo/configs/profile.yaml': 'base_config: ../../other/configs/base.yaml\n',
            'LightGenV2/tasks/other/configs/base.yaml': 'batch_size: 96\n',
        }
        self.assertEqual(set(snapshot(files).closure(['LightGenV2/tasks/demo/main.py'])), set(files))

    def test_unresolved_configuration_fails(self):
        with self.assertRaises(FileNotFoundError):
            snapshot({'LightGenV2/tasks/demo/configs/profile.yaml': 'base_config: missing.yaml\n'}).closure(
                ['LightGenV2/tasks/demo/configs/profile.yaml'])

    def test_runner_rejects_modified_source_and_does_not_initialize_git(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'source').mkdir()
            (root/'run_baseline.py').write_text(RUNNER, encoding='utf-8')
            (root/'TASK.json').write_text(json.dumps({'actions': {}}))
            content = b'original\n'
            (root/'source/model.py').write_bytes(content)
            (root/'SOURCE_MANIFEST.json').write_text(json.dumps({'source_commit': 'fixture', 'files': [
                {'path': 'model.py', 'sha256': hashlib.sha256(content).hexdigest()}]}))
            command = [sys.executable, str(root/'run_baseline.py'), 'check']
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertFalse((root/'source/.git').exists())
            (root/'source/model.py').write_bytes(b'modified\n')
            result = subprocess.run(command, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b'Source differs from snapshot', result.stderr)


if __name__ == '__main__': unittest.main()
