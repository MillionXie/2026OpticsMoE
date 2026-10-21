import ast,importlib.util,json
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parents[1]
class TerminalReopen(unittest.TestCase):
    def test_targets(self):
        p=HERE/'lab_caltech20_dvp_terminal_reopen_20261021.py'
        s=importlib.util.spec_from_file_location('offline',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
        self.assertEqual(m.target_count(346,411,1.5),340)
        self.assertEqual(m.target_count(337,411,1.2),333)
    def test_scope(self):
        c=json.loads((HERE/'lab_caltech20_L4_terminal_gap15_profile_20261021.json').read_text())
        self.assertEqual(c['tail']['epochs'],12);self.assertEqual(sum(c['original_TRAIN_counts']),3141)
        c2=json.loads((HERE/'lab_caltech20_L2_gap12_profile_20261021.json').read_text())
        self.assertEqual(c2['target_correct']['2'],333);self.assertFalse(c2['TEST_gradient'])
        source=(HERE/'lab_caltech20_dvp_terminal_reopen_20261021.py').read_text();ast.parse(source)
        self.assertNotIn('cuda()',source);self.assertNotIn('Bench(',source)
    def test_teacher_scope(self):
        c=json.loads((HERE/'lab_caltech20_L4_teacher_gap15_profile_20261021.json').read_text())
        self.assertEqual(c['method'],'TRAIN_teacher_ridge');self.assertEqual(c['parent_correct'],268)
        self.assertEqual(c['target'],340);self.assertNotIn('tail',c)
    def test_finite_continuation(self):
        c=json.loads((HERE/'lab_caltech20_L4_protected_gap15_profile_20261021.json').read_text())
        self.assertEqual(c['maximum_protected_passes'],3);self.assertEqual(c['parent_correct'],275)
        self.assertEqual(c['method'],'TRAIN_protected_continuation')
if __name__=='__main__':unittest.main()
