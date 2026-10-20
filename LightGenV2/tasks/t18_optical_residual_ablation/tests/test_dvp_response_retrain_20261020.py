import ast
from pathlib import Path
import unittest

SOURCE=Path(__file__).resolve().parents[1]/'lab_caltech20_dvp_response_retrain_20261020.py'


class RetrainTests(unittest.TestCase):
    def test_original_model_and_capture_not_replaced(self):
        source=SOURCE.read_text();tree=ast.parse(source)
        calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertNotIn('m.Bench',calls);self.assertNotIn('m.acquire',calls)
        self.assertIn("persistent=False",source)
        self.assertIn('set(model.state_dict())==set(base.state_dict())',source)
        self.assertIn("snapshot('best',0)",source)

    def test_surrogate_never_substitutes_native_gate(self):
        source=SOURCE.read_text()
        self.assertIn("eligible=native['correct']>=required_correct(a.depth,prefix)",source)
        self.assertIn('if eligible and m.improves',source)
        self.assertIn("results[name]=dict(TEST_development=score",source)
        self.assertIn("response_branch_deployment=False",source)

    def test_TEST_used_only_for_assessment(self):
        tree=ast.parse(SOURCE.read_text())
        loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and ast.unparse(n.target)=='begin' and ast.unparse(n.iter)=='range(0, 1600, 8)')
        self.assertNotIn("['test']",ast.unparse(loop))
        self.assertIn("['train']",ast.unparse(loop))

    def test_proxy_has_explicit_CPU_response_clone(self):
        source=SOURCE.read_text()
        self.assertIn('current=response_model(base,values,spatial,only_stage)',source)
        self.assertIn('current.response_enabled=True',source)
        self.assertIn('return m.assess(current,data,fields,\'test\')',source)

    def test_spatial_response_stays_at_verified_stage_and_freezes_head(self):
        source=SOURCE.read_text()
        self.assertIn("only_stage={4:'2',6:'0'}[a.depth]",source)
        self.assertIn('stage!=self.only_response_stage',source)
        self.assertIn("allowed=allowed-{'linear.weight','linear.bias'}",source)
        self.assertIn("dm['source_cache_SHA256']==m.SOURCE_CACHE",source)

    def test_native_anchor_teacher_is_TRAIN_only_and_detached(self):
        tree=ast.parse(SOURCE.read_text())
        teacher=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='TRAIN_teacher')
        text=ast.unparse(teacher)
        self.assertNotIn("['test']",text)
        self.assertIn("['train']",text)
        self.assertIn('torch.no_grad()',text)
        self.assertIn('not teacher.requires_grad',text)
        self.assertIn("teacher_TRAIN_SHA256",SOURCE.read_text())
        self.assertIn("assert not a.native_anchor or a.spatial_response is not None",SOURCE.read_text())

    def test_anchor_recipe_has_distinct_artifact_identity(self):
        tree=ast.parse(SOURCE.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='recipe_name');scope={}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<recipe>','exec'),scope)
        name=scope['recipe_name']
        self.assertEqual(name(4,True,True),'1_spatial_native_anchor_20261021')
        self.assertEqual(name(6,True,False),'router_spatial_response_retrain_20261020')
        with self.assertRaises(AssertionError):name(4,None,True)


if __name__=='__main__':unittest.main()
