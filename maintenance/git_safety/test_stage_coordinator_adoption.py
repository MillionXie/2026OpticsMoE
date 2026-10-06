"""Source-only tests; never execute a generated acquisition script."""
import ast
import hashlib
from pathlib import Path
import subprocess
import unittest

from LightGenV2.tasks.t06_video_quality_assessment.lab_stage_coordinator import desktop_code, STAGES

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'LightGenV2/tasks/t06_video_quality_assessment/lab_manual_stage.py'
PIN = '7093ec46082eed2fae127ec5028d3e2e8548b592'


class CoordinatorAdoptionTests(unittest.TestCase):
    def test_historical_function_preserved(self):
        old = subprocess.check_output(['git', 'show', PIN+':'+SOURCE], cwd=ROOT).decode('utf8')
        self.assertEqual(hashlib.sha256(old.encode()).hexdigest(),
                         'cd612caa81faa76f4e3ad38c985855b77d0737ead1424addbb52fbe5f59ced7d')
        current = (ROOT / SOURCE.replace('lab_manual_stage', 'lab_stage_coordinator')).read_text(encoding='utf8')
        def function(text):
            return ast.dump(next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'desktop_code'), include_attributes=False)
        self.assertEqual(function(old), function(current))

    def test_default_stages_match_runtime(self):
        tree = ast.parse((ROOT / 'LightGenV2/tasks/t06_video_quality_assessment/lab_runtime.py').read_text(encoding='utf8'))
        value = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'STAGES' for t in n.targets))
        self.assertEqual(STAGES, ast.literal_eval(value))

    def test_stage_helpers_preserved_and_existing_bench_unchanged(self):
        path = 'LightGenV2/tasks/t06_video_quality_assessment/lab_bench.py'
        old = ast.parse(subprocess.check_output(['git', 'show', PIN+':'+path], cwd=ROOT).decode('utf8'))
        before = ast.parse(subprocess.check_output(['git', 'show', 'd858b5b57f617650cdb0f1918875868b6d4ac7bf:'+path], cwd=ROOT).decode('utf8'))
        current = ast.parse((ROOT / path).read_text(encoding='utf8'))
        def definitions(tree):
            return {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        expected, prior, actual = definitions(old), definitions(before), definitions(current)
        for name in ('stage_config', 'effective_stage_identity'):
            self.assertEqual(expected[name], actual[name])
        for name, node in prior.items():
            self.assertEqual(node, actual[name], name)

    def test_generation_three_and_six_stages(self):
        for stages in (STAGES, STAGES[:3]):
            for stage in stages:
                for capture_only in (False, True):
                    code = desktop_code('p', 'b', 's', 'c', stage, 'out', stages, capture_only)
                    compile(code, '<generated-not-executed>', 'exec')
                    self.assertIn('Phase lease expired', code)
                    if len(stages) == 3:
                        self.assertNotIn('language_router', code)

    def test_module_has_no_top_level_calls_or_imports(self):
        path = ROOT / SOURCE.replace('lab_manual_stage', 'lab_stage_coordinator')
        tree = ast.parse(path.read_text(encoding='utf8'))
        self.assertFalse(any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in tree.body))
        self.assertTrue(all(isinstance(n, (ast.Expr, ast.Assign, ast.FunctionDef)) for n in tree.body))
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                self.assertFalse(any(isinstance(n, ast.Call) for n in ast.walk(node)))


if __name__ == '__main__':
    unittest.main()
