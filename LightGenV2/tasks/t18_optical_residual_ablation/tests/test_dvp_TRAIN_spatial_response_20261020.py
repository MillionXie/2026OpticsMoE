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
        self.assertIn('len(train[0])==len(held[0])==(98 if mask_review else 88)',source)

    def test_new_mask_response_is_audited_TRAIN_only_and_not_a_new_model(self):
        tree=ast.parse(SOURCE.read_text());loader=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='mask_review_rows')
        text=ast.unparse(loader)
        for token in ("report['response_frames'] == 60","not report['TEST_inference']","not report['stage_admission']","audit['passed']","range(20)","r['PT_SHA256'] == digest"):
            self.assertIn(token,text)
        for call in ('m.assess(','.backward(','.step(','.Bench('):self.assertNotIn(call,text)
        source=SOURCE.read_text()
        for token in ('bool(mask_review)==bool(spatial_prior)','torch.randperm(88)[:2]','88+torch.randperm(10)[:2]',"oldf['spatial_loss']<=1.05*oldb['spatial_loss']","newf['spatial_loss']<=.95*newb['spatial_loss']","newf['mean_coarse16_PCC']>=newb['mean_coarse16_PCC']+.03"):
            self.assertIn(token,source)

    def test_model_has_fixed_bounded_complexity(self):
        source=SOURCE.read_text()
        self.assertIn('parameters=27,phase_knots=17',source)
        self.assertIn('pupil_polynomials=8',source)
        self.assertIn('range(1,401)',source)
        self.assertIn('steps=F.softplus(increments)',source)


if __name__=='__main__':unittest.main()
