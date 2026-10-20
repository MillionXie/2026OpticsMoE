import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

HERE=Path(__file__).resolve().parents[1]


class ResponseRetakeTests(unittest.TestCase):
    def test_original_gate_required_and_failed_CCD_excluded(self):
        source=(HERE/'lab_caltech20_dvp_response_retake_20261020.py').read_text()
        self.assertIn('check_score(r,a.depth,prefix)',source)
        self.assertIn("assert not r['selected_e0']",source)
        self.assertIn("['router']+[str(i) for i in range(prefix-1)]",source)
        self.assertIn('failed_stage_CCD_reused=False',source)
        self.assertNotIn('shutil.copytree',source)

    def test_namespace_is_artifact_only_and_preserves_depth(self):
        tree=ast.parse((HERE/'lab_caltech20_dvp_layerwise_20261019.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='session_for');scope={}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<namespace>','exec'),scope)
        for depth in (4,6):
            a=SimpleNamespace(sequence=Path('run'),depth=depth,attempt='response_20261020')
            self.assertEqual(scope['session_for'](a),Path(f'run/L{depth}_response_20261020'))
        a.attempt='../../other'
        with self.assertRaises(AssertionError):scope['session_for'](a)

    def test_head_namespace_does_not_expand_native_or_pair_scope(self):
        tree=ast.parse((HERE/'lab_caltech20_dvp_conditioned_head_20261020.py').read_text())
        for name in ('native_terminal_admitted','independent_pair_admitted'):
            node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertNotIn('response_20261020',ast.unparse(node))
        self.assertIn("choices=('strict8pp_20261020','response_20261020')",(HERE/'lab_caltech20_dvp_conditioned_head_20261020.py').read_text())

    def test_compound_discloses_actual_last_not_fake_parent_best(self):
        text=(HERE/'lab_caltech20_dvp_response_retake_20261020.py').read_text()
        self.assertIn("node_protocol['parent_PT_SHA256']==m.sha(mask/'last.pt')",text)
        self.assertIn("actual_initialization_role='mask_last'",text)
        self.assertIn("assembly_optimizer_steps=0",text)
        self.assertIn("hr['selected_correct']>=required_correct(a.depth,prefix)",text)
        self.assertIn("if not k.startswith('linear.')",text)
        self.assertIn('head_continuation_chain=nodes',text)
        self.assertIn("range(4)",text)


if __name__=='__main__':unittest.main()
