"""Source-only contracts; never load weights, images, CUDA or optical SDKs."""
import ast
import hashlib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / 'LightGenV2/tasks/t12_text_to_image/perceptual'
ORIGINAL_SHA = {
    'eval_lpips.py': '952e3f4c5add32c41ebece215daa400f0c38580220bf739317208cb8e1a8bdb9',
    'eval_dists.py': 'ea0756841b95b0f26bf78cd69cfc7741e54397ad5cb8b48a5d7db66e827308ce',
    'eval_regions.py': 'd4462ef1888511d578184743cca3e799187bdcd0ae6ec03a9d8b6884ee37e0b6',
}


class MetricAdoptionTests(unittest.TestCase):
    def test_only_import_names_change(self):
        for name in ('eval_lpips.py', 'eval_dists.py', 'eval_regions.py'):
            restored = (TOOLS / name).read_text(encoding='utf8')
            old_module = 't12_eval_perceptual_20260928' if name == 'eval_regions.py' else 'eval_lpips'
            restored = restored.replace(
                'from LightGenV2.tasks.t12_text_to_image.perceptual.eval_lpips import mask_for, rgb, to_tensor',
                'from '+old_module+' import mask_for, rgb, to_tensor')
            self.assertEqual(hashlib.sha256(restored.encode('utf8')).hexdigest(), ORIGINAL_SHA[name])

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
