import ast
from pathlib import Path
import unittest
P=Path(__file__).with_name('run_corrected_boundaries_20261026.py')
class BoundaryContract(unittest.TestCase):
    def test_subtract_all_image_prepare_keep_tokenizer(self):
        t=ast.parse(P.read_text());n=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='accounted_ms');ns={};exec(compile(ast.Module(body=[n],type_ignores=[]),'test','exec'),ns)
        rows=[dict(CPU_start=0,CPU_end=2,H2D_end=3,tokenizer_intervals=[dict(start=1,end=1.5)]),dict(CPU_start=5,CPU_end=7,H2D_end=8,tokenizer_intervals=[dict(start=6,end=6.5)])]
        value,removed=ns['accounted_ms'](0,10,rows);self.assertEqual(removed,5000);self.assertEqual(value,5000)
    def test_default_Qwen_and_no_first_only_subtraction(self):
        s=P.read_text();self.assertIn('native.forward_batch(inputs,model,head,scores,timer)',s);self.assertIn('for row in rows:',s);self.assertIn('assert a>=metadata[-2]',s)
    def test_original_inputs_and_no_parallel_Qwen(self):
        s=P.read_text();self.assertIn('native.prepare_batch([row]',s);self.assertNotIn('ThreadPool',s);self.assertIn('no_graph=True,no_prefetch=True',s)
if __name__=='__main__':unittest.main()
