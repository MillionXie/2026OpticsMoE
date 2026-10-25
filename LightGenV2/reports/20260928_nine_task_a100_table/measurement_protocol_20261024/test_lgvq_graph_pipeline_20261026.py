import ast
from pathlib import Path
import unittest
from lgvq_graph_pipeline_20261026 import execution_plan


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
        self.assertIn('return native.forward(model,batch)',source)
        self.assertIn("static['pixels'].copy_(current['pixels']",source)
        self.assertIn("static['ids'].copy_(ids",source)
        self.assertIn('graph.replay()',source)
        self.assertNotIn('torch.compile',source)
        self.assertNotIn('make_graphed_callables(',source)
    def test_preserve_all_time_and_full_outputs(self):
        source=Path(__file__).with_name('lgvq_graph_pipeline_20261026.py').read_text(encoding='utf8')
        self.assertIn("elapsed=rows[-1]['end_ms']-begin",source)
        self.assertIn('first_eager_calls_and_capture_cost_in_continuous_window=True',source)
        self.assertIn('equal_tree(original,captured_output)',source)
        self.assertIn("assert args.system=='lg'",source)


if __name__=='__main__':unittest.main()
