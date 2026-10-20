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


if __name__=='__main__':unittest.main()
