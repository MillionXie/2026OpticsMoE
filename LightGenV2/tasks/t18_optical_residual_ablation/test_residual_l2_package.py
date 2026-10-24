"""No Torch/device requirement: package integrity and inference dependency closure."""
import ast
import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('t18_package', HERE / 'build_lab_package.py')
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


class PackageTest(unittest.TestCase):
    def test_syntax_and_no_device_imports(self):
        sources = package.SOURCES + ['lab_examples/residual_l2/example.py', 'build_lab_package.py']
        for name in sources:
            tree = ast.parse((HERE / name).read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self.assertFalse(any(x.name.startswith(('paramiko', 'ctypes', 'camera_')) for x in node.names))

    def test_built_archive(self):
        # Must run after committing this package's sources: only Git-authored code is exported.
        with tempfile.TemporaryDirectory(prefix='t18_l2_zip_test_') as folder:
            output = Path(folder) / 'reference.zip'
            package.build(output)
            with zipfile.ZipFile(output) as z:
                base = 'T18_L2_Residual_Simulation/'
                m = json.loads(z.read(base + 'manifest.json'))
                self.assertEqual(m['simulation_correct'], 337)
                self.assertFalse(m['includes_CCD'])
                for name, record in m['files'].items():
                    data = z.read(base + name)
                    self.assertEqual(package.digest(data), record['sha256'])
                    self.assertEqual(len(data), record['bytes'])
                self.assertFalse(any('ccd_raw' in x.lower() or 'credential' in x.lower() for x in z.namelist()))
                z.extractall(Path(folder) / 'isolated')
                entry = Path(folder) / 'isolated' / base / 'example.py'
                exec_spec = importlib.util.spec_from_file_location('t18_example_isolated', entry)
                module = importlib.util.module_from_spec(exec_spec)
                exec_spec.loader.exec_module(module)
                self.assertEqual(module.verify_files()['checkpoint_sha256'], package.PT_SHA)
            with self.assertRaises(AssertionError):
                package.build(output)


if __name__ == '__main__':
    unittest.main()
