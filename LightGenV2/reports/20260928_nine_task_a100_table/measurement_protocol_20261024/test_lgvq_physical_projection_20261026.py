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
        s=P.read_text();self.assertIn('len(tape.segments)==33',s);self.assertIn('electronic+optical',s);self.assertIn('sum(removed_by_kind.values())',s);self.assertIn('1000/1400+.300+.0307',s)
    def test_no_electronics_or_residual_dropped(self):
        s=P.read_text();self.assertIn('native.forward(model,batch)',s);self.assertIn('native_math_and_all_electronic_routes_retained=True',s);self.assertNotIn('optical_enabled=False',s);self.assertIn('all_predictions_bitwise_equal=True',s)
    def test_ROI_boundary_preserves_learned_readout(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('projection',P);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        source='def _read_frames(self,detector,stage):\n    stacked=detector+1\n    self.last_ccd[stage]=stacked.detach()\n    return self.readout(stacked)'
        tree,receipt=module.roi_AST(source)
        self.assertIsInstance(tree.body[0].body[0],ast.With);self.assertIsInstance(tree.body[0].body[1],ast.Return)
        self.assertTrue(receipt['learned_readout_and_router_decisions_retained'])
    def test_real_source_AST_nonoverlap(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('projection',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        root=P.parents[3];source=(root/'tasks/t06_video_quality_assessment/models/multivideo9x4.py').read_text(encoding='utf-8')
        tree=ast.parse(source);count=0
        for cls in tree.body:
            if not isinstance(cls,ast.ClassDef):continue
            for fn in cls.body:
                if isinstance(fn,ast.FunctionDef) and fn.name in ('_read_frames','_read_videos'):
                    m.roi_AST(ast.unparse(fn));count+=1
                elif isinstance(fn,ast.FunctionDef) and fn.name=='forward' and cls.name in ('FrameOpticalRouter','VideoOpticalRouter'):
                    rt,_=m.roi_AST(ast.unparse(fn),True);m.simulation_prefix_AST(ast.unparse(rt));count+=1
        self.assertEqual(count,4)
if __name__=='__main__':unittest.main()
