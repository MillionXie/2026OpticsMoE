import ast
from pathlib import Path
import unittest
P=Path(__file__).with_name('lgvq_physical_projection_20261026.py')
class ProjectionContract(unittest.TestCase):
    def test_prefix_preserved_and_post_detector_retained(self):
        tree=ast.parse(P.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='simulation_prefix_AST');ns=dict(ast=ast,__import__=__import__)
        import textwrap,hashlib
        ns.update(textwrap=textwrap,hashlib=hashlib);exec(compile(ast.Module(body=[fn],type_ignores=[]),'test','exec'),ns)
        source='def expert(self,x):\n    y=x+1\n    detector,guard=self._detector(y)\n    return self.readout(detector),guard'
        result,receipt=ns['simulation_prefix_AST'](source)
        self.assertEqual(receipt['prefix_statements'],2);self.assertTrue(receipt['original_prefix_math_unchanged']);self.assertIsInstance(result.body[0].body[0],ast.With);self.assertIsInstance(result.body[0].body[1],ast.Return)
    def test_all_six_and_no_double_count(self):
        s=P.read_text();self.assertIn('len(tape.segments)==13',s);self.assertIn('electronic+optical',s);self.assertIn('full-removed_h2d-removed_book-sim-sim_capture',s);self.assertIn('1000/1400+.300+.0307',s)
    def test_no_electronics_or_residual_dropped(self):
        s=P.read_text();self.assertIn('native.forward(model,batch)',s);self.assertIn('native_math_and_all_electronic_routes_retained=True',s);self.assertNotIn('optical_enabled=False',s);self.assertIn('all_predictions_bitwise_equal=True',s)
if __name__=='__main__':unittest.main()
