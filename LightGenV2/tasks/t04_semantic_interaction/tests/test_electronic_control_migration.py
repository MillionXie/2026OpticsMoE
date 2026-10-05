"""CPU source contracts; never imports Torch, datasets or checkpoints."""
import argparse
import ast
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'benchmark_electronic_control.py'


def functions():
    tree = ast.parse(SOURCE.read_text(encoding='utf8'))
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    for node in selected:
        node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *selected], type_ignores=[])
    ns = {'Path': Path, 'argparse': argparse}
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), 'exec'), ns)
    return ns


class ElectronicControlMigrationTests(unittest.TestCase):
    def setUp(self):
        self.ns = functions()

    def load(self, state, optical=False):
        value = SimpleNamespace(dtype='fake', to=lambda **kwargs: value)
        model = SimpleNamespace(state_dict=lambda: {'weight': value},
                                load_state_dict=lambda s, strict: self.assertEqual((set(s), strict), ({'weight'}, True)))
        model.eval = lambda: model
        model.requires_grad_ = lambda enabled: model
        self.ns.update(load_settings=lambda p: SimpleNamespace(optical_enabled=optical, output_dir=Path('run')),
                       build_model=lambda s, d: model,
                       torch=SimpleNamespace(load=lambda *a, **k: state))
        return self.ns['_load_selected_model'](Path('config'), None, None)

    def test_ema_only_checkpoint_loads_strictly(self):
        value = SimpleNamespace(dtype='fake')
        value.to = lambda **kwargs: value
        self.load({'ema_model': {'weight': value}})

    def test_missing_or_extra_keys_are_rejected(self):
        for state in [{}, {'weight': None, 'extra': None}]:
            with self.subTest(keys=list(state)), self.assertRaises(RuntimeError):
                self.load({'model': state})

    def test_optical_config_is_not_electronic_control(self):
        with self.assertRaises(RuntimeError):
            self.load({}, optical=True)

    def test_existing_output_rejected_before_device_access(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                self.ns['run'](argparse.Namespace(output_dir=Path(directory)))

    def test_invalid_options_rejected_before_device_access(self):
        with tempfile.TemporaryDirectory() as directory:
            for samples, gpu in [(0, 'A100'), (1000, ' ')]:
                with self.subTest(samples=samples, gpu=gpu), self.assertRaises(ValueError):
                    self.ns['run'](argparse.Namespace(output_dir=Path(directory)/'unused', samples=samples, expected_gpu=gpu))

    def test_cli_keeps_historical_defaults(self):
        captured = []
        self.ns['run'] = captured.append
        with patch('sys.argv', ['control', '--config', 'c', '--output-dir', 'o']):
            self.assertEqual(self.ns['main'](), 0)
        self.assertEqual((captured[0].samples, captured[0].expected_gpu), (1000, 'A100'))

    def test_measurement_scope_and_physical_gpu_are_explicit(self):
        text = SOURCE.read_text(encoding='utf8')
        for contract in ['"not_full_qwen3vl_baseline": True', '"explicit_warmup_forwards": 0',
                         '"first_test_sample_included": True', 'NvidiaSmiPowerSampler(gpu_index=physical_gpu)',
                         'power_limit_w=actual_power_limit', 'Qwen instruction-hidden precomputation']:
            self.assertIn(contract, text)


if __name__ == '__main__':
    unittest.main()
