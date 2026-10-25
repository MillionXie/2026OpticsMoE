"""Small scheduling/source safety checks; no torch, models, video or GPU."""
import ast
from pathlib import Path
import unittest
import lgvq_pipeline_demo_20261025 as demo


class DemoContract(unittest.TestCase):
    def test_complete_fields_and_padding(self):
        plan=demo.field_plan(40)
        self.assertEqual(len(plan),3)
        self.assertEqual(sum(sum(flags) for _,flags in plan),40)
        self.assertTrue(all(len(ids)==len(flags)==16 for ids,flags in plan))
        self.assertEqual(plan[-1][0],list(range(32,40))+list(range(8)))
        self.assertEqual(plan[-1][1],[True]*8+[False]*8)
    def test_ordered_valid_ids(self):
        for n in (16,32,40,64):
            actual=[i for ids,flags in demo.field_plan(n) for i,v in zip(ids,flags) if v]
            self.assertEqual(actual,list(range(n)))
    def test_bounded_demo(self):
        for n in (0,15,65,200):
            with self.assertRaises(ValueError):demo.field_plan(n)
    def test_no_module_level_model_import(self):
        tree=ast.parse(Path(demo.__file__).read_text(encoding='utf8'))
        imports=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
        self.assertFalse(any('torch' in ast.unparse(n) or 'transformers' in ast.unparse(n) for n in imports))
    def test_timing_not_named_physical_end_to_end(self):
        source=Path(demo.__file__).read_text(encoding='utf8')
        self.assertIn('physical_optical_E2E_not_measured=True',source)
        self.assertIn('projection_excludes_CPU_and_is_NOT_E2E=True',source)
        self.assertIn('full_TEST_not_repeated=True,Qwen_not_executed=True',source)
    def test_stage_meter_has_no_per_scope_drain(self):
        tree=ast.parse(Path(demo.__file__).read_text(encoding='utf8'))
        node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EventMeter')
        self.assertNotIn('synchronize(',ast.unparse(node))


if __name__=='__main__':unittest.main()
