"""No-Torch protocol tests for historical same-weight ablation migration."""
import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def functions(filename, names):
    tree = ast.parse((ROOT / filename).read_text(encoding='utf8'))
    selected = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            selected.append(node)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *selected], type_ignores=[])
    ns = {'Path': Path, 'argparse': argparse, 'json': json}
    exec(compile(ast.fix_missing_locations(module), filename, 'exec'), ns)
    return ns


def core():
    value = SimpleNamespace(fusion_ablation_mode='none')
    value.set_fusion_ablation = lambda mode: setattr(value, 'fusion_ablation_mode', mode)
    return value


class FusionAblationMigrationTests(unittest.TestCase):
    def setUp(self):
        self.ns = functions('training.py', {'_set_fusion_ablation', 'evaluate_selected', 'evaluate_with_routes'})

    def test_both_cores_change_and_reset_together(self):
        model = SimpleNamespace(language_core=core(), vision_core=core())
        for mode in ('remove_optical', 'remove_electronic', 'none'):
            self.ns['_set_fusion_ablation'](model, mode)
            self.assertEqual((model.language_core.fusion_ablation_mode, model.vision_core.fusion_ablation_mode), (mode, mode))

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            self.ns['_set_fusion_ablation'](None, 'invalid')

    def test_baseline_has_no_fusion_to_ablate(self):
        self.ns['_set_fusion_ablation'](SimpleNamespace(), 'none')
        with self.assertRaises(ValueError):
            self.ns['_set_fusion_ablation'](SimpleNamespace(), 'remove_optical')

    def test_partial_support_does_not_mutate_one_core(self):
        first = core()
        with self.assertRaises(ValueError):
            self.ns['_set_fusion_ablation'](SimpleNamespace(language_core=first), 'remove_optical')
        self.assertEqual(first.fusion_ablation_mode, 'none')

    def test_cli_keeps_none_default_and_profiles(self):
        ns = functions('run.py', {'main'})
        profiles = {'main_dc20': 'x', 'qwen_shared': 'y', 'embedding_alpha40': 'z'}
        args = []
        ns.update(PROFILES=profiles, PHASES={'evaluate'}, run=lambda a: args.append(a) or {})
        with patch('sys.argv', ['run', '--profile', 'main_dc20', '--phase', 'evaluate']), patch('builtins.print'):
            ns['main']()
        self.assertEqual(args[0].fusion_ablation, 'none')

    def test_ablation_cannot_trigger_training_or_preparation(self):
        ns = functions('run.py', {'run'})
        for phase in ('all', 'train', 'prepare'):
            with self.subTest(phase=phase), self.assertRaises(ValueError):
                ns['run'](argparse.Namespace(fusion_ablation='remove_optical', phase=phase))

    def test_existing_reports_and_predictions_rejected_before_data_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode in ('none', 'remove_optical', 'remove_electronic'):
                suffix = '' if mode == 'none' else '_' + mode
                for name in ('selected_checkpoint_test_evaluation' + suffix + '.json', 'test_predictions' + suffix + '.jsonl'):
                    path = root / name
                    path.touch()
                    with self.subTest(name=name), self.assertRaises(FileExistsError):
                        self.ns['evaluate_selected'](SimpleNamespace(output_dir=root), None, Path('pt'), mode)
                    path.unlink()

    def test_cli_existing_report_guard_precedes_metadata_or_device(self):
        ns = functions('run.py', {'run'})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'selected_checkpoint_test_evaluation_remove_optical.json').touch()
            ns.update(TASK_DIR=root, PROFILES={'main_dc20': 'c'}, load_settings=lambda p: SimpleNamespace(output_dir=root))
            with self.assertRaises(FileExistsError):
                ns['run'](argparse.Namespace(profile='main_dc20', run_dir=None, phase='evaluate', fusion_ablation='remove_optical'))

    def test_remove_optical_skips_shared_router_hooks(self):
        sentinel = object()
        self.ns['legacy'] = SimpleNamespace(_evaluate=lambda *args: sentinel)
        model = SimpleNamespace(router_backend='optical', _optics_are_ablated=lambda: True)
        self.assertIs(self.ns['evaluate_with_routes'](model, None, SimpleNamespace(shared_readout_enabled=True), None), sentinel)

    def test_same_checkpoint_strict_load_and_separate_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode in ('none', 'remove_optical', 'remove_electronic'):
                paths = []
                loaded = []
                model = SimpleNamespace(language_core=core(), vision_core=core(), router_backend='none', checkpoint_architecture='contract',
                                        load_state_dict=lambda s, strict: loaded.append((s, strict)), eval=lambda: None, architecture_report=lambda: {})
                state = {'preserved': 'tensors'}
                self.ns.update(build_loaders=lambda s: (None, object()), build_model=lambda s, d: model,
                               torch=SimpleNamespace(load=lambda *a, **k: {'architecture': 'contract', 'model': state, 'epoch': 7}),
                               _set_phase_dropout=lambda *a: None, _json=lambda p, value: paths.append(p),
                               evaluate_with_routes=lambda *a: ({'overall': {'changed_cell_accuracy': .5}}, [], []),
                               legacy=SimpleNamespace(_save_gallery=lambda p, *a: paths.append(p)))
                result = self.ns['evaluate_selected'](SimpleNamespace(output_dir=root, test_samples=1000, embedding_only=False), None, root/'fixed.pt', mode)
                self.assertEqual(loaded, [(state, True)])
                self.assertEqual((result['fusion_ablation'], result['selected_epoch']), (mode, 7))
                suffix = '' if mode == 'none' else '_' + mode
                self.assertTrue((root/('test_predictions'+suffix+'.jsonl')).exists())
                self.assertIn(root/('selected_checkpoint_test_evaluation'+suffix+'.json'), paths)
                self.assertIn(root/('best_visualization'+suffix)/'test_examples', paths)

    def test_model_ablation_detection_is_modality_aware(self):
        ns = functions('modeling.py', {'_optics_are_ablated'})
        model = SimpleNamespace(language_core=core(), vision_core=core())
        self.assertFalse(ns['_optics_are_ablated'](model))
        model.vision_core.set_fusion_ablation('remove_optical')
        self.assertTrue(ns['_optics_are_ablated'](model))


if __name__ == '__main__':
    unittest.main()
