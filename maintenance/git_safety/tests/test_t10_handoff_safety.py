"""Synthetic reporting fixtures only; no formal T10 data/checkpoints or TEST execution."""
import ast
import contextlib
import hashlib
import importlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
PREFIX = 'LightGenV2.tasks.t10_expert_scaling.'
NAMES = ('export_topk32_handoff', 'update_topk32_test_handoff', 'update_topk32_full_test_handoff')
guard = importlib.import_module(PREFIX + 'handoff_output')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf8')


def snapshot(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


def fixture(source, full):
    module = importlib.import_module(PREFIX + ('update_topk32_full_test_handoff' if full else 'update_topk32_test_handoff'))
    selected = {4: 3, 16: 16, 25: 12, 49: 24}
    grid = {4: (1, 2, 3, 4), 16: (1, 4, 8, 16), 25: (1, 6, 12, 25), 49: (1, 12, 24, 49)}
    specs, training = [], []
    for n, ks in grid.items():
        for architecture, values in [('moe_oeo', ks), ('d2nn_expert_global', (n,))]:
            for k in values:
                for seed in (17, 27):
                    name = f'{architecture}_{n}_{k}_{seed}'
                    spec = dict(run_name=name, architecture=architecture, experts=n, top_k=k,
                                seed=seed, checkpoint_sha256='synthetic-pt-' + name)
                    training.append(dict(run_name=name, architecture=architecture, experts=n,
                                         top_k=k, seed=seed, val_accuracy=.7, val_macro_f1=.7))
                    if full or architecture != 'moe_oeo' or k == selected[n]:
                        specs.append(spec)
    write_json(source / 'evidence/dataset/data_manifest.json', {'classes': ['fixture0', 'fixture1']})
    write_json(source / 'plotting_summary.json', {'completion': {}})
    module.write_csv(source / 'per_seed_runs.csv', training)
    lockpath = source / ('full_test_scan_lock.json' if full else 'test_selection_lock.json')
    write_json(lockpath, {'runs': specs, 'test_policy': 'synthetic fixture, no model execution'})
    lockhash = hashlib.sha256(lockpath.read_bytes()).hexdigest()
    for spec in specs:
        n, k = spec['experts'], spec['top_k']
        values = dict(accuracy=.75 if spec['architecture'] == 'moe_oeo' else .7,
                      balanced_accuracy=.7, macro_f1=.7, macro_nll=.5, capture_mean=.1,
                      confusion_matrix=[[300, 76], [76, 300]])
        if spec['architecture'] == 'moe_oeo':
            values.update(route_probability=[1 / n] * n, route_load=[k / n] * n, distinct_selected_sets=1)
        write_json(source / 'evidence/runs' / spec['run_name'] / 'test_result.json',
                   dict(state='complete', selection_lock_sha256=lockhash,
                        checkpoint_sha256=spec['checkpoint_sha256'], test_samples=752,
                        data_sha256='synthetic-data', test_ids_sha256='synthetic-ids', test=values))
    return module, specs


class HandoffTests(unittest.TestCase):
    def test_existing_output_and_input_tree_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            write_json(source / 'original.json', {'keep': True})
            before = snapshot(source)
            with self.assertRaises(FileExistsError):
                guard.prepare_output(source, source, [])
            with self.assertRaises(ValueError):
                guard.prepare_output(source, source / 'new', [])
            self.assertEqual(before, snapshot(source))

    def test_missing_evidence_creates_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                guard.prepare_output(root / 'source', root / 'new', ['absent.json'])
            self.assertFalse((root / 'new').exists())

    def test_all_cli_require_explicit_output(self):
        for name in NAMES:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                importlib.import_module(PREFIX + name).main(['--package', 'unused-fixture'])
            self.assertEqual(error.exception.code, 2)

    def test_original_math_helpers_unchanged(self):
        for name in NAMES:
            path = 'LightGenV2/tasks/t10_expert_scaling/' + name + '.py'
            original = subprocess.check_output(['git', '-C', str(ROOT), 'show',
                '417a90397fb1eb7cf49f138b69c752ddc5a31afd:' + path]).decode('utf8')
            def functions(text):
                return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(text).body
                        if isinstance(n, ast.FunctionDef) and n.name != 'main'}
            self.assertEqual(functions(original), functions((ROOT / path).read_text(encoding='utf8')))

    def test_full_scan_reports_without_modifying_input(self):
        self.run_report(True)

    def test_locked_subset_keeps_unselected_test_cells_empty(self):
        self.run_report(False)

    def run_report(self, full):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'original'
            source.mkdir()
            module, specs = fixture(source, full)
            before = snapshot(source)
            output = root / 'derived'
            with contextlib.redirect_stdout(io.StringIO()):
                module.main(['--package', str(source), '--output', str(output)])
            self.assertEqual(before, snapshot(source))
            rows = module.read_csv(output / 'per_seed_runs.csv')
            self.assertEqual(len(rows), 40)
            self.assertEqual(sum(row['test_evaluated'] == 'True' for row in rows), len(specs))
            if not full:
                self.assertTrue(all(row['test_accuracy'] == '' for row in rows if row['test_evaluated'] == 'False'))
            self.assertEqual(len(module.read_csv(output / 'full_test_per_seed.csv' if full else output / 'locked_test_per_seed.csv')), len(specs))
            report = json.loads((output / 'plotting_summary.json').read_text())
            self.assertEqual(report['completion']['test_evaluated_runs'], len(specs))
            comparison = module.read_csv(output / ('topk_test_trends.csv' if full else 'best_topk_vs_d2nn.csv'))
            self.assertEqual(len(comparison), 16 if full else 4)
            for row in comparison:
                self.assertAlmostEqual(float(row['moe_test_accuracy_mean']), .75)
                self.assertAlmostEqual(float(row['d2nn_test_accuracy_mean']), .7)
                self.assertAlmostEqual(float(row['moe_minus_d2nn_test_pp']), 5)
                self.assertEqual(float(row['moe_test_accuracy_sd']), 0)
            with self.assertRaises(FileExistsError):
                module.main(['--package', str(source), '--output', str(output)])


if __name__ == '__main__':
    unittest.main()
