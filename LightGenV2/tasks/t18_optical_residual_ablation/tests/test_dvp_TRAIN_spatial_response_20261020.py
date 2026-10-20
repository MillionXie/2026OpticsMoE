import ast
from pathlib import Path
import unittest

SOURCE=Path(__file__).resolve().parents[1]/'lab_caltech20_dvp_TRAIN_spatial_response_20261020.py'


class SpatialResponseTests(unittest.TestCase):
    def test_no_TEST_PT_SDK_or_coordinate_mutation(self):
        source=SOURCE.read_text();tree=ast.parse(source)
        calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
        for name in ('m.data_for','m.assess','torch.save','m.Bench','m.acquire','F.grid_sample','cv2.warpPerspective'):
            self.assertNotIn(name,calls)
        self.assertIn('LUT_written=False',source)
        self.assertIn('physical_phase_calibrated=False',source)

    def test_held_spatial_guard_not_just_brightness(self):
        source=SOURCE.read_text()
        self.assertIn("final['held_TRAIN']['mean_coarse16_PCC']>=baseline['held_TRAIN']['mean_coarse16_PCC']+.03",source)
        self.assertIn("v['index']%2==0",source);self.assertIn("v['index']%2==1",source)
        self.assertIn('len(train[0])==len(held[0])==88',source)

    def test_model_has_fixed_bounded_complexity(self):
        source=SOURCE.read_text()
        self.assertIn('parameters=27,phase_knots=17',source)
        self.assertIn('pupil_polynomials=8',source)
        self.assertIn('range(1,401)',source)
        self.assertIn('steps=F.softplus(increments)',source)


if __name__=='__main__':unittest.main()
