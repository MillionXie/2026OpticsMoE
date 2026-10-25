import ast
from pathlib import Path
import unittest
from lgvq_graph_pipeline_20261026 import execution_plan,constant_allocation_AST,deferred_diagnostics_AST


class GraphContract(unittest.TestCase):
    def test_first_real_calls_are_retained_not_dummy(self):
        plan=execution_plan()
        self.assertEqual(len(plan),128);self.assertEqual(plan[:3],['eager_real']*3)
        self.assertEqual(plan.count('graph_real'),125)
        self.assertEqual((plan.count('eager_real')+plan.count('graph_real'))*16,2048)
    def test_short_plan_rejected(self):
        with self.assertRaises(AssertionError):execution_plan(3)
    def test_original_complete_forward_and_copy_every_replay(self):
        source=Path(__file__).with_name('lgvq_graph_pipeline_20261026.py').read_text(encoding='utf8');ast.parse(source)
        self.assertIn('task=native.forward(model,batch)',source)
        self.assertIn("static['pixels'].copy_(current['pixels']",source)
        self.assertIn("static['ids'].copy_(ids",source)
        self.assertIn('graph.replay()',source)
        self.assertNotIn('torch.compile',source)
        self.assertNotIn('make_graphed_callables(',source)
    def test_preserve_all_time_and_full_outputs(self):
        source=Path(__file__).with_name('lgvq_graph_pipeline_20261026.py').read_text(encoding='utf8')
        self.assertIn("elapsed=rows[-1]['end_ms']-begin",source)
        self.assertIn('first_eager_calls_and_capture_cost_in_continuous_window=True',source)
        self.assertIn("equal_tree(original,captured_output['task'])",source)
        self.assertIn("assert args.system=='lg'",source)
    def test_only_constant_creation_replaced(self):
        source='def f(raw,settings):\n    x=raw.new_tensor(settings.value)\n    return x*raw+3'
        tree=constant_allocation_AST(source)
        self.assertIn('_graph_constant(raw, settings.value)',ast.unparse(tree))
        self.assertEqual(ast.dump(tree.body[0].body[-1]),ast.dump(ast.parse(source).body[0].body[-1]))
    def test_unknown_constant_pattern_rejected(self):
        with self.assertRaises(AssertionError):constant_allocation_AST('def f(raw):\n    return raw+3')
    def test_only_diagnostic_float_removed(self):
        source='def forward(self,x):\n    y=x*3\n    self.last_diagnostics={"a":float(x),"b":float(x+1),"c":float(y),"d":float(y+1)}\n    return y+5'
        tree=deferred_diagnostics_AST(source)
        self.assertEqual(ast.dump(tree.body[0].body[0]),ast.dump(ast.parse(source).body[0].body[0]))
        self.assertEqual(ast.dump(tree.body[0].body[-1]),ast.dump(ast.parse(source).body[0].body[-1]))
        self.assertNotIn('float(',ast.unparse(tree))
    def test_no_unrecognized_diagnostic_rewrite(self):
        with self.assertRaises(AssertionError):deferred_diagnostics_AST('def forward(self,x):\n    return x')


if __name__=='__main__':unittest.main()
