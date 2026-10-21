import ast
from pathlib import Path
import unittest

SOURCE=Path(__file__).resolve().parents[1]/'lab_caltech20_dvp_restored_confusion_20261021.py'


class CoupledConfusion(unittest.TestCase):
    def test_finite_same_original_head(self):
        text=SOURCE.read_text();ast.parse(text)
        for token in ("cfg['fit_seconds']==120", "cfg['max_pairs']==20", "gradient_batch=1600", "assert (weight>0).all()",
            "zip(pair,(v,-v))", "other18_rows_fixed_per_pair=True", "TEST_gradient=False", "new_CCD=0", "new_inference_layers=0",
            "loop.validate_state(m,model,original,{'linear.weight','linear.bias'})"):
            self.assertIn(token,text)
        for bad in ('.cuda(','.Bench(','.capture(','.settings('):self.assertNotIn(bad,text)
    def test_derivative_only_train_arguments(self):
        tree=ast.parse(SOURCE.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='direction')
        self.assertEqual([n.arg for n in fn.args.args],['v','x','z0','y','teacher','keep','pair','weight','c'])
        self.assertNotIn('test',ast.unparse(fn).lower())
    def test_original_replay_and_numeric_hessian(self):
        text=SOURCE.read_text();self.assertIn('torch.autograd.functional.hessian(loss,v)',text)
        self.assertIn("loop.original_replay(model,data,fields,'train')",text)
        self.assertIn('error<=3e-6',text)


if __name__=='__main__':unittest.main()
