import importlib.util
from pathlib import Path
import unittest
import numpy as np

SOURCE=Path(__file__).resolve().parents[1]/'lab_caltech20_dvp_TRAIN_response_20261020.py'
spec=importlib.util.spec_from_file_location('TRAIN_response',SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class ResponseTests(unittest.TestCase):
    def test_nominal_copy_and_immutable_phase(self):
        x=np.arange(478*478,dtype=np.uint8).reshape(478,478)
        original=x.copy();rows=m.phase_patterns(x)
        self.assertEqual([r[0] for r in rows],['nominal_before','flat','half_code','global_offset128','nominal_after'])
        np.testing.assert_array_equal(rows[0][1],x)
        np.testing.assert_array_equal(rows[-1][1],x)
        np.testing.assert_array_equal(rows[2][1],x//2)
        np.testing.assert_array_equal(rows[3][1],((x.astype(np.uint16)+128)%256).astype(np.uint8))
        rows[0][1][0,0]=99
        np.testing.assert_array_equal(x,original)

    def test_checker_is_bounded_diagnostic_only(self):
        for value in (0,64,128,192):
            x=m.checker_phase(value)
            self.assertEqual(x.shape,(478,478));self.assertEqual(x.dtype,np.uint8)
            self.assertTrue(set(np.unique(x))<={0,value})
        with self.assertRaises(AssertionError):m.checker_phase(255)

    def test_no_TEST_loader_or_optimization_or_setting_mutation(self):
        import ast
        tree=ast.parse(SOURCE.read_text())
        calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
        for banned in ('m.data_for','m.fields_for','m.fit','m.assess','m.torch.save','bench.camera.settings'):
            self.assertNotIn(banned,calls)
        self.assertFalse(any('optim.' in name for name in calls))
        self.assertIn('bench.stage',calls)
        self.assertIn('m.shared_geometry_check',calls)

    def test_exact_original_capture_parent_cases(self):
        self.assertEqual([(r[0],r[1]) for r in m.CASES],[(4,'2'),(6,'0')])
        self.assertTrue(all(len(r[3])==64 for r in m.CASES))


if __name__=='__main__':unittest.main()
