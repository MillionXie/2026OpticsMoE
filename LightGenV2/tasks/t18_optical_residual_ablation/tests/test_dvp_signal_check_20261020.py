"""Scope only: original published marker check does the numerical validation."""
import ast
from pathlib import Path
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'lab_caltech20_dvp_signal_check_20261020.py'


class SignalCheckScope(unittest.TestCase):
    def test_fixed_scope_and_settings(self):
        code = SOURCE.read_text()
        self.assertIn("cfg['exposure_us'] == 60006.0", code)
        self.assertIn("cfg['gain'] == 1.0", code)
        self.assertIn('range(2)', code)
        self.assertIn('known_patterns=4', code)
        self.assertIn('m.shared_geometry_check(', code)
        self.assertNotIn('SetExposure', code)
        self.assertNotIn('data_for(', code)
        self.assertNotIn('.cuda(', code)

    def test_release_and_no_retry(self):
        tree = ast.parse(SOURCE.read_text())
        run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
        self.assertTrue(any(isinstance(n, ast.Try) and n.finalbody for n in ast.walk(run)))
        self.assertIn('m.LOCK.unlink()', SOURCE.read_text())
        self.assertIn('automatic_retry=False', SOURCE.read_text())


if __name__ == '__main__':
    unittest.main()
