"""Pure AST/scalar contract tests; do not import Torch, data or a model."""

import ast
import inspect
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lgvq_scoreonly_20261026 as candidate

MODEL = HERE.parents[3]/'LightGenV2/tasks/t06_video_quality_assessment/models/multivideo9x4.py'


def original_functions():
    tree = ast.parse(MODEL.read_text(encoding='utf-8'))
    model = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                 and node.name == 'MultiVideo9x4OpticalVQA')
    forward = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == 'forward')
    router = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_standardized_router')
    return ast.unparse(forward), ast.unparse(router)


class ScoreOnlyContract(unittest.TestCase):
    def test_real_forward_has_only_allowlisted_auxiliary_removals(self):
        source, _ = original_functions()
        tree, receipt = candidate.score_forward_AST(source)
        self.assertEqual(receipt['removed_statement_counts'], dict(alignments=4, guard_losses=4, initializers=2))
        transformed = tree.body[0]
        original = ast.parse(source).body[0]
        # Every remaining call in the scoring prefix has exactly the same AST
        # and original order. Only four _alignment calls may disappear.
        def calls(function):
            result = []
            ignored = set()
            for node in ast.walk(function):
                if not isinstance(node, ast.Expr) or not isinstance(node.value,ast.Call):
                    continue
                call = node.value
                if (isinstance(call.func, ast.Attribute) and isinstance(call.func.value,ast.Name)
                        and call.func.value.id in ('alignments','guard_losses')):
                    ignored.update(id(child) for child in ast.walk(node))
            for node in ast.walk(function):
                if not isinstance(node, ast.Call) or id(node) in ignored:
                    continue
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id in ('alignments','guard_losses'):
                    continue
                if isinstance(node.func, ast.Name) and node.func.id == '_alignment':
                    continue
                result.append(candidate._dump(node))
            return result
        end = next(i+1 for i,node in enumerate(original.body) if isinstance(node, ast.Assign)
                   and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='prediction')
        original.body = original.body[:end]
        transformed.body = transformed.body[:-1]
        self.assertEqual(calls(original), calls(transformed))
        self.assertFalse(receipt['complete_auxiliary_dictionary_unchanged'])
        self.assertTrue(receipt['score_math_unchanged'])

    def test_router_preserves_softmax_top2_and_retained_values_exactly(self):
        _, source = original_functions()
        original = ast.parse(source).body[0]
        tree, receipt = candidate.router_score_AST(source)
        new = tree.body[0]
        self.assertEqual(candidate._dump(ast.Module(body=original.body[:-1],type_ignores=[])),
                         candidate._dump(ast.Module(body=new.body[:-1],type_ignores=[])))
        before = {k.value:candidate._dump(v) for k,v in zip(original.body[-1].value.keys,original.body[-1].value.values) if k is not None}
        after = {k.value:candidate._dump(v) for k,v in zip(new.body[-1].value.keys,new.body[-1].value.values)}
        self.assertEqual(before, after)
        self.assertTrue(receipt['probabilities_softmax_top2_weights_unchanged'])

    def test_reject_unknown_auxiliary_router_expansion(self):
        _, source = original_functions()
        with self.assertRaises(AssertionError):
            candidate.router_score_AST(source.replace('_slotwise_routing_statistics','unapproved_other_work'))

    def test_reject_changed_auxiliary_shape(self):
        source, _ = original_functions()
        with self.assertRaises(AssertionError):
            candidate.score_forward_AST(source.replace('alignments.append(_alignment','alignments.append(other_function',1))

    def test_worker_scoped_hook_restores_on_failure(self):
        original_install = candidate.projection.install_simulation_boundaries
        def fail(*args):
            self.assertIsNot(candidate.projection.install_simulation_boundaries, original_install)
            raise RuntimeError('fake original worker failure')
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output=Path(directory))
            with patch.object(candidate.lowlevel, 'worker', fail):
                with self.assertRaisesRegex(RuntimeError, 'fake original worker failure'):
                    candidate.worker(args, object(), 2048)
            self.assertIs(candidate.projection.install_simulation_boundaries, original_install)

    def test_success_relabels_new_leaf_honestly(self):
        def fake_worker(args, guard, N):
            for name in ('projection_identity.json','lowlevel_capture_identity.json','timing.json'):
                (args.output/name).write_text(json.dumps(dict(original_model_math_unchanged=True,
                    original_scopes_and_math_unchanged=True, auxiliary_outputs_changed=False)),encoding='utf-8')
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output=Path(directory))
            with patch.object(candidate.lowlevel, 'worker', fake_worker):
                candidate.worker(args, object(), 2048)
            for name in ('projection_identity.json','lowlevel_capture_identity.json','timing.json'):
                output = json.loads((args.output/name).read_text())
                self.assertNotIn('original_model_math_unchanged',output)
                self.assertNotIn('original_scopes_and_math_unchanged',output)
                self.assertTrue(output['score_math_unchanged'])
                self.assertTrue(output['auxiliary_outputs_omitted'])
                self.assertTrue(output['auxiliary_outputs_changed'])
                self.assertFalse(output['complete_auxiliary_dictionary_unchanged'])
                self.assertTrue(output['fusion_four_diagnostic_tensors_retained'])

    def test_no_fusion_or_prediction_reference_replacement(self):
        source = Path(candidate.__file__).read_text(encoding='utf-8')
        self.assertIn('lowlevel.worker(args, guard, N)', source)
        self.assertNotIn('fusion.forward =', source)
        self.assertNotIn('optical_enabled=False', source)
        self.assertNotIn('np.load(', source)

    def test_thin_runner_freezes_all_new_and_original_sources(self):
        path = HERE/'run_lgvq_scoreonly_20261026.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        called = []
        runner = SimpleNamespace(entry=SimpleNamespace(REPO=HERE.parents[3],
            base=SimpleNamespace(sha=lambda p:'sha-'+p.name)), main=lambda argv:called.append(argv))
        namespace = dict(Path=Path, runner=runner, scoreonly=candidate,
            THIS_ENTRY=path, ORIGINAL_THIN_ENTRY=HERE/'run_lgvq_lowlevel_20261026.py',
            PREVIOUS_SOURCES=lambda:{'retained.py':'sha'})
        exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),namespace)
        namespace['main'](['--phase','suite'])
        frozen = namespace['sources']()
        self.assertEqual(called,[['--phase','suite']])
        for filename in ('run_lgvq_scoreonly_20261026.py','run_lgvq_lowlevel_20261026.py',
                         'lgvq_scoreonly_20261026.py','lgvq_lowlevel_capture_20261026.py',
                         'lgvq_physical_projection_20261026.py'):
            self.assertIn((HERE/filename).relative_to(runner.entry.REPO).as_posix(),frozen)


if __name__ == '__main__':
    unittest.main()
