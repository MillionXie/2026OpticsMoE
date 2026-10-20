"""Pure policy/namespace tests: no Torch, camera, training or experiment writes."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('strict_retake', HERE / 'lab_caltech20_dvp_strict_retake_20261020.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def report(correct, **changes):
    value = dict(state='complete', frozen_parameters_unchanged=True, selected_correct=correct,
                 results=dict(best=dict(strict_CPU=True, PT_SHA256='pt',
                     TEST_development=dict(correct=correct, samples=411))))
    value.update(changes)
    return value


class StrictRetakeTests(unittest.TestCase):
    def test_integer_targets(self):
        for depth, middle, final in ((4, 314, 342), (6, 322, 350)):
            for prefix in range(1, depth + 1):
                self.assertEqual(module.required_correct(depth, prefix), middle)
            self.assertEqual(module.required_correct(depth, depth + 1), final)

    def test_below_target_never_carries(self):
        for depth, prefix, score in ((4, 3, 311), (4, 4, 297), (6, 1, 318), (6, 2, 303)):
            with self.assertRaises(AssertionError):
                module.check_score(report(score, within8pp=True, finite_budget_progression_already_in_admitted_profile=True), depth, prefix)

    def test_exact_target_passes(self):
        for depth, prefix, score in ((4, 2, 314), (6, 1, 322), (4, 5, 342), (6, 7, 350)):
            self.assertEqual(module.check_score(report(score), depth, prefix), 'pt')

    def test_final_is_not_middle_gate(self):
        with self.assertRaises(AssertionError):
            module.check_score(report(314), 4, 5)

    def test_full_native_population_required(self):
        q = report(330)
        q['results']['best']['TEST_development']['samples'] = 40
        with self.assertRaises(AssertionError):
            module.check_score(q, 4, 3)
        q = report(330)
        q['results']['best']['strict_CPU'] = False
        with self.assertRaises(AssertionError):
            module.check_score(q, 4, 3)

    def test_namespace_is_opt_in_and_depth_not_changed(self):
        tree = ast.parse((HERE / 'lab_caltech20_dvp_layerwise_20261019.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'session_for')
        scope = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<namespace>', 'exec'), scope)
        a = SimpleNamespace(sequence=Path('sequence'), depth=4)
        self.assertEqual(scope['session_for'](a), Path('sequence/L4'))
        a.attempt = 'strict8pp_20261020'
        self.assertEqual(scope['session_for'](a), Path('sequence/L4_strict8pp_20261020'))
        a.attempt = '../L2'
        with self.assertRaises(AssertionError):
            scope['session_for'](a)

    def test_original_forward_and_fit_math_not_changed(self):
        tree = ast.parse((HERE / 'lab_caltech20_dvp_layerwise_20261019.py').read_text())
        acquire = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'acquire')
        self.assertIn('check_score', ast.unparse(acquire))
        self.assertNotIn('carry_p', ast.unparse(acquire))

    def test_whole_white_is_not_an_ordinary_signal_extremum(self):
        text=(HERE / 'lab_caltech20_dvp_layerwise_20261019.py').read_text()
        self.assertIn("if row['saturation_fraction']>.99 else 0", text)
        self.assertIn('if white_run>=2:', text)
        self.assertIn('finish_pending();raise RuntimeError(\'Whole-frame white event', text)
        self.assertIn("if getattr(a,'attempt',None):", text)

    def test_retained_prefix_is_sha_bound_and_train_only(self):
        text=(HERE / 'lab_caltech20_dvp_layerwise_20261019.py').read_text()
        self.assertIn("quality_source and role=='train' and index<quality_count", text)
        self.assertIn("qr['prefix_SHA256']==identities", text)
        self.assertIn("qr['hardware_SHA256']==hardware_digest", text)
        self.assertIn("row['input_SHA256']==sha(bmp)", text)
        self.assertIn('pcc>=.95 and .8<=ratio<=1.2', text)
        self.assertIn('new_physical_capture=False', text)
        self.assertIn('reused_verified_TRAIN_prefix=quality_count', text)


if __name__ == '__main__':
    unittest.main()
