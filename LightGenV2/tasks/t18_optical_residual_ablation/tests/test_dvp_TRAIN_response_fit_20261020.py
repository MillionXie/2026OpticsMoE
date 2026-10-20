import ast
from pathlib import Path
import unittest

SOURCE=Path(__file__).resolve().parents[1]/'lab_caltech20_dvp_TRAIN_response_fit_20261020.py'


class ResponseFitTests(unittest.TestCase):
    def test_source_does_not_load_TEST_or_modify_PT_or_SDK(self):
        tree=ast.parse(SOURCE.read_text());calls=[ast.unparse(v.func) for v in ast.walk(tree) if isinstance(v,ast.Call)]
        for forbidden in ('m.data_for','m.fields_for','m.assess','torch.save','m.Bench','m.fit'):
            self.assertNotIn(forbidden,calls)
        self.assertNotIn('bench.camera.settings',calls)

    def test_fit_holdout_are_disjoint_and_not_dataset_VALID(self):
        source=SOURCE.read_text()
        self.assertIn("v['index']%2==0",source);self.assertIn("v['index']%2==1",source)
        self.assertIn("len(training)==len(held)==88",source)
        self.assertIn("final['held_TRAIN']<=baseline['held_TRAIN']*.95",source)

    def test_finite_and_repeat_guards(self):
        source=SOURCE.read_text()
        self.assertIn("r['coarse_PCC']>=.95",source)
        self.assertIn("torch.isfinite(raw.grad).all()",source)
        self.assertIn("TEST_labels_used=False",source)
        self.assertIn("physical_LUT_calibrated=False",source)


if __name__=='__main__':unittest.main()
