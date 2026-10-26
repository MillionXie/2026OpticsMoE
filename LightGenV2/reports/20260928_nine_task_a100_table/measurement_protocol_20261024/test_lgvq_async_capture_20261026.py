"""Zero-model checks for the candidate's real-field capture plan/AST route."""
import ast
import copy
from contextlib import nullcontext
import hashlib
from pathlib import Path
from types import SimpleNamespace
import textwrap
import unittest

import lgvq_async_capture_20261026 as capture


class CaptureContractTests(unittest.TestCase):
    def test_real_slot_capture_replays_each_graph_once_and_restores(self):
        from test_lgvq_async_replay_20261026 import Tensor, Event, Stream, Graph
        tapes = []; restored = []
        class Tape:
            def __init__(self, torch, stream):
                self.stream = stream; self.context = None; self.segments = []; self.capturing = False
                tapes.append(self)
            def start(self, kind, label):
                self.context = (kind, label)
            def close(self):
                kind, label = self.context
                self.segments.append(dict(kind=kind, label=label, graph=Graph()))
                self.context = None
            def boundary(self, kind):
                self.close(); self.start(kind, kind); self.close(); self.start('electronics', 'after_'+kind)
        torch = SimpleNamespace(empty_like=lambda t: Tensor(), is_tensor=lambda t: isinstance(t, Tensor),
            cuda=SimpleNamespace(Event=Event, stream=lambda s: nullcontext(), synchronize=lambda: None))
        def tail(prefix):
            for kind in ['optical_simulation']*6+['detector_ROI']*6+['electronic_residual']*4:
                tapes[0].boundary(kind)
            return dict(prediction=Tensor(19))
        def install(model, tape):
            return lambda: restored.append(True), []
        raw = {k: Tensor() for k in ('pixels', 'quality', 'ids', 'mask')}
        slot, prediction, record = capture.capture_real_slot(torch, field=3, raw=raw,
            cpu_ids=Tensor(), cpu_mask=Tensor(), model=object(), front_stream=Stream(), body_stream=Stream(),
            prepare_prefix=lambda static: dict(vision=Tensor(), prompt_tokens=Tensor()), forward_tail=tail,
            tape_factory=Tape, install_boundaries=install)
        self.assertEqual(restored, [True])
        self.assertEqual(prediction.value, 19)
        self.assertEqual(len(slot['segments']), 34)
        self.assertTrue(all(s['graph'].calls == 1 for s in slot['segments']))
        self.assertEqual(len(record['events']), 36)
        self.assertEqual(slot['score_ready'].waits, 1)
        self.assertIsNone(tapes[0].context)

    def test_actual_score_tail_is_retained_without_auxiliary_losses(self):
        import lgvq_scoreonly_20261026 as score
        here = Path(capture.__file__).parent
        model = ast.parse((here.parents[3]/'LightGenV2/tasks/t06_video_quality_assessment/models/multivideo9x4.py').read_text(encoding='utf-8'))
        cls = next(n for n in model.body if isinstance(n, ast.ClassDef) and n.name == 'MultiVideo9x4OpticalVQA')
        forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
        score_tree, _ = score.score_forward_AST(ast.unparse(forward))
        helper = ast.parse((here/'lgvq_hoisted_pipeline_20261026.py').read_text(encoding='utf-8'))
        fn = next(n for n in helper.body if isinstance(n, ast.FunctionDef) and n.name == 'partition_forward')
        namespace = dict(ast=ast, copy=copy, hashlib=hashlib, textwrap=textwrap,
                         PREFIX_NAMES=('batch', 'videos', 'vision', 'prompt_tokens', 'prompt_mask'))
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<original_partition>', 'exec'), namespace)
        partitioned, receipt = namespace['partition_forward'](ast.unparse(score_tree))
        self.assertTrue(receipt['original_tail_AST_unchanged'])
        self.assertEqual([n.name for n in partitioned.body], ['prepare_native_prefix', 'forward_from_prefix'])
        forbidden = {'_alignment', 'alignments', 'guard_losses'}
        self.assertFalse(any(isinstance(n, ast.Name) and n.id in forbidden for n in ast.walk(partitioned)))
        keys = {k.value for k in partitioned.body[1].body[-1].value.keys}
        self.assertIn('prediction', keys)
        self.assertFalse(set(score.OMITTED_OUTPUTS) & keys)

    def test_all_128_real_fields_retained(self):
        plan = capture.serial_capture_plan()
        self.assertEqual(len(plan), 128)
        self.assertEqual(plan[:5], ['eager_real']*3+['capture_slot0_real', 'capture_slot1_real'])
        self.assertEqual(plan.count('async_replay_real'), 123)

    def test_no_small_discardable_probe(self):
        for fields in [0, 3, 5]:
            with self.assertRaises(ValueError):
                capture.serial_capture_plan(fields)

    def test_import_never_loads_torch_or_models(self):
        tree = ast.parse(Path(capture.__file__).read_text(encoding='utf-8'))
        imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        names = [a.name for n in imports if isinstance(n, ast.Import) for a in n.names]
        self.assertEqual(names, ['ast', 'inspect', 'time'])

    def test_partition_uses_score_ast_then_tail_partition(self):
        calls = []
        def forward(self):
            return None
        def score(source):
            calls.append('score')
            return ast.parse('def forward(self):\n    return 7'), dict(approved=True)
        def router(source):
            calls.append('router')
            return ast.parse('def _standardized_router():\n    return 8'), {}
        def partition(source):
            calls.append('partition')
            self.assertIn('return 7', source)
            return ast.parse('def prepare_native_prefix(self):\n    return 1\ndef forward_from_prefix(self):\n    return 2'), {}
        forward.__globals__['_standardized_router'] = forward
        model = type('Model', (), {'forward': forward})()
        prefix, tail, receipt = capture.scoring_partition(
            model, SimpleNamespace(score_forward_AST=score, router_score_AST=router),
            SimpleNamespace(partition_forward=partition))
        self.assertEqual(calls, ['score', 'partition', 'router'])
        self.assertEqual((prefix(model), tail(model)), (1, 2))
        self.assertTrue(receipt['auxiliary_outputs_omitted'])


if __name__ == '__main__':
    unittest.main()
