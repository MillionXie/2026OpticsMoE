import ast,json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
class PhysicalDirectionContract(unittest.TestCase):
    def test_full_TRAIN_evidence_and_single_opposite(self):
        text=(ROOT/'lab_caltech20_dvp_closedloop_sign_20261021.py').read_text();ast.parse(text)
        for token in ("'minimum_TRAIN_NLL_worsening'","'minimum_TRAIN_correct_loss'",'new_TRAIN_gradient_records=0','TRAIN_basis_records=1600','global_incumbent_correct=global_correct','loop.acquire(','loop.refine(',"returned['returncode']==0"):
            self.assertIn(token,text)
        for token in ('.Bench(','.backward(','.step(',"settings(exposure",'TEST_gradient=True'):self.assertNotIn(token,text)
        profile=json.loads((ROOT/'lab_caltech20_DVP_closedloop_sign_profile_20261021.json').read_text());self.assertEqual(profile['maximum_trials_each_depth'],1)
    def test_no_new_inference_parameters(self):
        text=(ROOT/'lab_caltech20_dvp_closedloop_sign_20261021.py').read_text()
        self.assertIn('validate_state(m,model,before,keys)',text);self.assertIn('mirrored_state(before,pos,keys)',text)

if __name__=='__main__':unittest.main()
