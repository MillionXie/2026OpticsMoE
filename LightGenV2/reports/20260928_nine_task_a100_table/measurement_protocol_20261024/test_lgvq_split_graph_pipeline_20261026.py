import ast
from pathlib import Path
import unittest
P=Path(__file__).with_name('lgvq_split_graph_pipeline_20261026.py')

class SplitContract(unittest.TestCase):
    def test_buffer_lifetime(self):
        tree=ast.parse(P.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='buffer_for');namespace={};exec(compile(ast.Module(body=[node],type_ignores=[]),'buffer','exec'),namespace)
        self.assertEqual([namespace['buffer_for'](i) for i in range(3,9)],[0,1,0,1,0,1])
        with self.assertRaises(AssertionError):namespace['buffer_for'](2)
    def test_original_math_and_two_dependencies(self):
        s=P.read_text();self.assertIn('h.original_functions(model)',s);self.assertIn("fs.wait_event(slot['released'])",s);self.assertIn('bs.wait_event(front_end)',s)
        self.assertIn("target['task']=tail_fn(model,target['prefix'],optical_enabled=True)",s)
    def test_complete_outputs_and_cost_retained(self):
        s=P.read_text();self.assertIn('graphbase.execution_plan()',s);self.assertIn('first_eager_calls_and_capture_cost_in_continuous_window=True',s);self.assertIn('equal_tree(original',s);self.assertIn('np.array_equal(expected,diagnostics[-1])',s)
    def test_not_prediction_cache(self):
        s=P.read_text();self.assertIn("slot['front_graph'].replay()",s);self.assertIn("slot['body_graph'].replay()",s);self.assertIn("slot['static'][key].copy_(current[key]",s);self.assertIn('source',s)

if __name__=='__main__':unittest.main()
