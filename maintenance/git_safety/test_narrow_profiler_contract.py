"""Test historical timing identities without importing Torch or running a GPU."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


class Placeholder:
    def __init__(self, *args, **kwargs):
        pass

    def to(self, *args):
        return self

    def eval(self):
        return self

    def __setitem__(self, key, value):
        pass


class TimingIdentityTest(unittest.TestCase):
    def test_legacy_and_historical_contracts_are_separate(self):
        path = Path(__file__).resolve().parents[2] / 'LightGenV2/scripts/profile_narrow_optical_electronics_a100.py'
        tree = ast.parse(path.read_text(encoding='utf8'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'add_standard')
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), function], type_ignores=[])
        rows, heads, kernels = [], [], []

        def head(kind):
            heads.append(kind)
            return Placeholder()

        def residual(*args, **kwargs):
            kernels.append(kwargs['kernel_size'])
            return Placeholder()

        torch = SimpleNamespace(rand=Placeholder, randn=Placeholder, ones=Placeholder, zeros=Placeholder, bool=bool)
        legacy = SimpleNamespace(SPECS={'t08': SimpleNamespace(vision_tokens=196, language=True, language_tokens=77, router_passes=2)}, StandardCCDReadout=Placeholder)
        scope = dict(torch=torch, full=SimpleNamespace(legacy=legacy, AboI2IHead=head, AboI2IResidual=residual), benchmark=lambda *args, **kwargs: rows.append(kwargs['shape_contract']))
        exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), scope)
        call = scope['add_standard']
        call('abo_image_to_image', 't08', 'synthetic', trial=1, warmup=0, repeats=0)
        self.assertEqual(heads, ['linear64'])
        self.assertEqual(kernels, [3, 5])
        self.assertIn('[1,77,192]', rows[-1])
        rows.clear(); heads.clear(); kernels.clear()
        call('abo_image_to_image', 't08', 'synthetic', trial=1, warmup=0, repeats=0, abo_i2i_contract='20260927_71_spatial64')
        self.assertEqual(heads, ['spatial2x2_64'])
        self.assertEqual(kernels, [5, 5])
        self.assertIn('[1,71,192]', rows[-1])
        with self.assertRaises(ValueError):
            call('abo_image_to_image', 't08', 'synthetic', trial=1, warmup=0, repeats=0, abo_i2i_contract='rank72')


if __name__ == '__main__':
    unittest.main()
