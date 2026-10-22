import ast,json,unittest
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parents[1]
SOURCE=HERE/'lab_caltech20_dvp_layer_diagnosis_20261022.py'


class StageDiagnosis(unittest.TestCase):
    def test_no_device_or_training_operations(self):
        source=SOURCE.read_text();ast.parse(source)
        for text in ('Bench(', 'Camera(', '.backward(', 'optim.', 'advance.acquire(', 'dvpStart('):self.assertNotIn(text,source)
        self.assertIn('return observed',source)
        self.assertIn("np.array_equal(gray,np.asarray(Image.open(p)))",source)
        self.assertIn("matched=={k:2011 for k in paths}",source)
        self.assertIn("all(torch.equal(v,m.state_cpu(model)[k])",source)
    def test_probe_excludes_same_source_and_TEST_fit(self):
        tree=ast.parse(SOURCE.read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='frozen_geometry')
        scope={'np':np};exec(compile(ast.Module(body=[function],type_ignores=[]),'diagnostic_math','exec'),scope)
        fn=scope['frozen_geometry'];rng=np.random.default_rng(73);y=np.arange(80)%20;x=np.eye(20)[y]+rng.normal(0,.01,(80,20));ids=np.array([str(i) for i in range(80)])
        a=fn(x,y,ids,x,y);b=fn(x,y,ids,x,np.roll(y,1))
        self.assertEqual(a['TRAIN_leave_source_out_probe_correct'],80)
        self.assertEqual(a['TRAIN_nearest_same_class_correct'],80)
        for key in ('TRAIN_probe_correct','TRAIN_leave_source_out_probe_correct','probe_regularization','rank95'):
            self.assertEqual(a[key],b[key])
        # Duplicates share a source; the left-out statistic excludes both.
        duplicate_ids=np.concatenate((ids, np.array([v+'::TRAIN_shift_dx+2_dy-1' for v in ids])))
        c=fn(np.tile(x,(2,1)),np.tile(y,2),duplicate_ids,x,y)
        self.assertTrue(c['same_source_excluded']);self.assertEqual(c['TRAIN_leave_source_out_probe_correct'],160)
    def test_fixed_protocol(self):
        cfg=json.loads((HERE/'lab_caltech20_L4_layer_diagnosis_profile_20261022.json').read_text())
        self.assertEqual(cfg['TRAIN_nominal_samples'],200);self.assertEqual(cfg['probe_penalty_fraction'],.1)
        self.assertLessEqual(cfg['maximum_seconds'],1200)


if __name__=='__main__':unittest.main()
