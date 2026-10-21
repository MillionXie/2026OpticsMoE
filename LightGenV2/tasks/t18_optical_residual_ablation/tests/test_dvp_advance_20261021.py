"""Static/protocol tests run without requiring local Torch or hardware."""
import ast
import importlib.util
import json
from pathlib import Path
import unittest

TASK=Path(__file__).resolve().parents[1]
SOURCE=TASK/'lab_caltech20_dvp_advance_20261021.py'


class AdvanceTests(unittest.TestCase):
    def test_stage_order(self):
        spec=importlib.util.spec_from_file_location('advance',SOURCE)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertEqual(module.schedule(4),['3'])
        self.assertEqual(module.schedule(6),['1','2','3','4','5'])

    def test_config_scope(self):
        c=json.loads((TASK/'lab_caltech20_DVP_advance_profile_20261021.json').read_text())
        self.assertTrue(c['user_current_stage_waiver'])
        self.assertFalse(c['repeat_current_stage_fit'])
        self.assertFalse(c['TEST_gradient'])
        self.assertEqual(c['VALID'],0)
        self.assertEqual(c['parents']['4']['correct'],285)
        self.assertEqual(c['parents']['6']['correct'],287)
        self.assertEqual(c['parents']['6']['prefix'],2)

    def test_capture_and_fit_guards(self):
        s=SOURCE.read_text();ast.parse(s)
        self.assertIn("with m.LOCK.open('x')",s)
        self.assertIn("for role in ('train','test')",s)
        self.assertIn('loop.audit_capture(',s)
        self.assertIn('transforms.remove()',s)
        self.assertIn("assert m.sha(parent)==c['SHA256']",s)
        self.assertNotIn('propose(',s)


if __name__=='__main__':unittest.main()
