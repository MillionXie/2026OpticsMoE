"""CPU-only historical selection tests; no training, Torch or real PT I/O."""
import ast
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
BACKEND = ROOT/'experiments/qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing/training.py'
ENTRY = Path(__file__).resolve().parents[1]/'electronic_control.py'


def functions(path, names):
    nodes = [n for n in ast.parse(path.read_text(encoding='utf8')).body if isinstance(n, ast.FunctionDef) and n.name in names]
    for n in nodes:
        n.decorator_list = []
    tree = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    ns = {'Path': Path}
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), ns)
    return ns


class ElectronicControlSelectionTests(unittest.TestCase):
    def setUp(self):
        self.ns = functions(BACKEND, {'selected_checkpoint_path', 'train', '_load_official'})

    def test_legacy_default_does_not_select_test(self):
        self.assertEqual(self.ns['selected_checkpoint_path'](SimpleNamespace(output_dir=Path('r'))), Path('r/checkpoints/last.pt'))

    def test_test_selection_is_explicit_without_last_fallback(self):
        self.assertEqual(self.ns['selected_checkpoint_path'](SimpleNamespace(output_dir=Path('r')), 'test_development'), Path('r/checkpoints/best_test_changed.pt'))
        with self.assertRaises(ValueError):
            self.ns['selected_checkpoint_path'](SimpleNamespace(output_dir=Path('r')), 'invalid')

    def test_train_guards_existing_artifacts_before_seed_or_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('checkpoints/last.pt', 'checkpoints/best_train_loss.pt', 'checkpoints/best_test_changed.pt', 'training_summary.json', 'train_log.csv', 'test_log.csv'):
                path = root/name
                path.parent.mkdir(exist_ok=True)
                path.touch()
                with self.subTest(name=name), self.assertRaises(FileExistsError):
                    self.ns['train'](SimpleNamespace(output_dir=root, resume=False), None, checkpoint_selection='test_development')
                path.unlink()

    def test_maximum_score_updates_and_ties_keep_first(self):
        train = next(n for n in ast.parse(BACKEND.read_text(encoding='utf8')).body if isinstance(n, ast.FunctionDef) and n.name == 'train')
        block = next(n for n in ast.walk(train) if isinstance(n, ast.If) and "float(overall['changed_cell_accuracy']) > best_test_changed" in ast.unparse(n.test))
        code = compile(ast.fix_missing_locations(ast.Module(body=[block], type_ignores=[])), '<selection-block>', 'exec')
        saved = []
        ns = dict(checkpoint_selection='test_development', best_test_changed=float('-inf'), best_test_epoch=-1,
                  _checkpoint=lambda *args: saved.append(args[5]), selected_checkpoint_path=lambda *a: None,
                  settings=None, model=None, optimizer=None, scheduler=None, ema=None, global_step=0, row={})
        for epoch, score in enumerate([.3, .6, .4, .6], 1):
            ns.update(epoch=epoch, overall={'changed_cell_accuracy': score})
            exec(code, ns)
        self.assertEqual((saved, ns['best_test_epoch'], ns['best_test_changed']), ([1, 2], 2, .6))
        ns.update(checkpoint_selection='last_epoch', epoch=5, overall={'changed_cell_accuracy': .9})
        exec(code, ns)
        self.assertEqual(saved, [1, 2])

    def test_default_training_and_loading_signatures_remain_legacy(self):
        for name in ('train', '_load_official'):
            self.assertEqual(self.ns[name].__kwdefaults__['checkpoint_selection'], 'last_epoch')

    def test_selected_checkpoint_missing_fails_before_model_build(self):
        seen = []
        def load(path, **kwargs):
            seen.append(path)
            raise FileNotFoundError(path)
        self.ns['torch'] = SimpleNamespace(load=load)
        with self.assertRaises(FileNotFoundError):
            self.ns['_load_official'](SimpleNamespace(output_dir=Path('r')), None, checkpoint_selection='test_development')
        self.assertEqual(seen, [Path('r/checkpoints/best_test_changed.pt')])

    def test_control_entry_rejects_optical_model_before_device(self):
        ns = functions(ENTRY, {'run'})
        ns['load_settings'] = lambda p: SimpleNamespace(optical_enabled=True)
        with self.assertRaises(ValueError):
            ns['run'](SimpleNamespace(config=Path('c')))

    def test_control_entry_always_selects_historical_development_pt(self):
        ns = functions(ENTRY, {'run'})
        with tempfile.TemporaryDirectory() as directory:
            settings = SimpleNamespace(optical_enabled=False, output_dir=Path(directory))
            called = []
            ns.update(load_settings=lambda p: settings, torch=SimpleNamespace(device=lambda d: d),
                      train=lambda s, d, **k: called.append(('train', k)), test=lambda s, d, **k: called.append(('test', k)))
            for phase in ('train', 'evaluate'):
                ns['run'](SimpleNamespace(config=Path('c'), run_dir=None, device='cpu', phase=phase))
            self.assertEqual(called, [('train', {'checkpoint_selection': 'test_development'}), ('test', {'checkpoint_selection': 'test_development'})])


if __name__ == '__main__':
    unittest.main()
