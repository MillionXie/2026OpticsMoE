import ast
from pathlib import Path
import unittest

SOURCE=Path(__file__).resolve().parents[1]/'lab_caltech20_dvp_TRAIN_mask_review_20261021.py'

class TRAINMaskReviewContract(unittest.TestCase):
    def test_exact_three_cases_and_no_learning(self):
        tree=ast.parse(SOURCE.read_text());env={}
        nodes=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CASES' for t in n.targets)]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'<cases>','exec'),env)
        a,b,c=env['CASES'];self.assertEqual(len(env['CASES']),3);self.assertEqual(a[1:],c[1:]);self.assertNotEqual(a[3],b[3])
        text=SOURCE.read_text()
        for banned in ('torch.save(','.backward(','.step(','.cuda(',"settings(exposure",'check_score('):self.assertNotIn(banned,text)
        for token in ('stage_admission=False','next_layer_capture=False','TEST_inference=False','new_PT_written=False','automatic_retry=False'):
            self.assertIn(token,text)

    def test_reuses_original_physical_prefix_and_canonical_encoding(self):
        text=SOURCE.read_text()
        for token in ("m.allowed_keys(model,3)","m.fields_for(session,3",'m.layer_forward(model,x,ids',"np.rint(amplitude[:n].clip(0,1)*255)","m.native(drive[gray],'amplitude')",'m.phase_native(',"finally:bench.stage(nominal)","assert white<2"):
            self.assertIn(token,text)
        self.assertIn("cfg['exposure_us']==60006.",text);self.assertIn("cfg['orientation']=='flip_v'",text)
        tree=ast.parse(text);audit=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='audit')
        self.assertNotIn('Bench(',ast.unparse(audit))

if __name__=='__main__':unittest.main()
