"""Source-only contracts; never load weights, images, CUDA or optical SDKs."""
import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / 'LightGenV2/tasks/t12_text_to_image/perceptual'
OLD = ROOT / 'outputs/t12_perceptual_comparison_20260928'


class MetricAdoptionTests(unittest.TestCase):
    def test_only_import_names_change(self):
        for name in ('eval_lpips.py', 'eval_dists.py', 'eval_regions.py'):
            expected = (OLD / name).read_text(encoding='utf8')
            for module in ('eval_lpips', 't12_eval_perceptual_20260928'):
                expected = expected.replace('from '+module+' import mask_for, rgb, to_tensor',
                    'from LightGenV2.tasks.t12_text_to_image.perceptual.eval_lpips import mask_for, rgb, to_tensor')
            self.assertEqual(expected.rstrip(), (TOOLS / name).read_text(encoding='utf8').rstrip())

    def test_local_helper_import_resolves_to_defined_symbols(self):
        helper = ast.parse((TOOLS / 'eval_lpips.py').read_text(encoding='utf8'))
        defined = {n.name for n in helper.body if isinstance(n, ast.FunctionDef)}
        for name in ('eval_dists.py', 'eval_regions.py'):
            tree = ast.parse((TOOLS / name).read_text(encoding='utf8'))
            imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                       and n.module == 'LightGenV2.tasks.t12_text_to_image.perceptual.eval_lpips']
            self.assertEqual(len(imports), 1)
            self.assertTrue({n.name for n in imports[0].names} <= defined)

    def test_execution_guard_and_existing_output_rejection(self):
        for name in ('eval_lpips.py', 'eval_dists.py', 'eval_regions.py'):
            source = (TOOLS / name).read_text(encoding='utf8')
            tree = ast.parse(source)
            self.assertIn('if __name__ == "__main__":', source)
            self.assertIn('if args.output.exists():', source)
            calls = [n for n in tree.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)]
            self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
