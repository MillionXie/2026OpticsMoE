"""Pure fake-CUDA contracts: no Torch import, model, CUDA or input data."""

import argparse
import ast
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lgvq_lowlevel_capture_20261026 as candidate


class FakeCUDA:
    def __init__(self, fail_begin=False):
        self.log = []
        self.current = 'outside'
        self.graphs = []
        self.fail_begin = fail_begin

    def synchronize(self):
        self.log.append('synchronize')

    def empty_cache(self):
        self.log.append('empty_cache')

    def stream(self, stream):
        cuda = self
        class Context:
            def __enter__(self):
                self.previous = cuda.current
                cuda.current = stream
                cuda.log.append(('enter', stream))
            def __exit__(self, *error):
                cuda.log.append(('exit', stream))
                cuda.current = self.previous
        return Context()

    def CUDAGraph(self):
        cuda = self
        class Graph:
            def capture_begin(self, **kwargs):
                self.stream = cuda.current
                self.kwargs = kwargs
                cuda.log.append(('begin', self.stream, kwargs))
                if cuda.fail_begin:
                    raise RuntimeError('fake capture failure')
            def capture_end(self):
                assert cuda.current == self.stream
                cuda.log.append(('end', self.stream))
        graph = Graph()
        self.graphs.append(graph)
        return graph


class LowLevelContract(unittest.TestCase):
    def tape(self, **kwargs):
        cuda = FakeCUDA(**kwargs)
        return candidate.LowLevelTape(SimpleNamespace(cuda=cuda), 'worker_stream'), cuda

    def test_33_segments_one_preparation_and_private_pools(self):
        tape, cuda = self.tape()
        tape.capturing = True
        with patch.object(candidate.gc, 'collect') as collect:
            tape.start('electronics', 'frontend')
            kinds = ['optical_simulation']*6 + ['detector_ROI']*6 + ['electronic_residual']*4
            for index, kind in enumerate(kinds):
                with tape.excluded(str(index), kind):
                    pass
            tape.close()
        self.assertEqual(len(tape.segments), 33)
        self.assertEqual(sum(s['kind']=='electronics' for s in tape.segments), 17)
        self.assertEqual(tape.preparation_calls, 1)
        self.assertEqual(cuda.log.count('synchronize'), 1)
        self.assertEqual(cuda.log.count('empty_cache'), 1)
        collect.assert_called_once_with()
        self.assertEqual(cuda.current, 'outside')
        for graph in cuda.graphs:
            self.assertEqual(graph.stream, 'worker_stream')
            self.assertEqual(graph.kwargs, {'capture_error_mode': 'global'})
        self.assertTrue(all(s['independent_default_pool'] for s in tape.segments))
        self.assertGreaterEqual(tape.segments[0]['capture_host_ms'], tape.preparation['host_ms'])
        self.assertTrue(tape.preparation['charged_in_first_electronics_segment'])

    def test_failed_begin_restores_stream(self):
        tape, cuda = self.tape(fail_begin=True)
        with patch.object(candidate.gc, 'collect'):
            with self.assertRaisesRegex(RuntimeError, 'fake capture failure'):
                tape.start('electronics', 'frontend')
        self.assertIsNone(tape.context)
        self.assertEqual(cuda.current, 'outside')

    def test_rejects_nested_capture_and_uncharged_initial_preparation(self):
        tape, _ = self.tape()
        with self.assertRaisesRegex(RuntimeError, 'must be charged'):
            tape.start('optical_simulation', 'not_first_electronics')
        with patch.object(candidate.gc, 'collect'):
            tape.start('electronics', 'frontend')
            with self.assertRaisesRegex(RuntimeError, 'Cannot nest'):
                tape.start('electronics', 'overwrite')
            tape.close()

    def test_worker_reuses_original_and_restores_class(self):
        original_tape = candidate.projection.Tape
        def original_worker(args, guard, N):
            self.assertIs(candidate.projection.Tape, candidate.LowLevelTape)
            self.assertEqual(N, 2048)
            tape, _ = self.tape()
            tape.capturing = True
            tape.start('electronics', 'frontend')
            for kind in ['optical_simulation']*6 + ['detector_ROI']*6 + ['electronic_residual']*4:
                with tape.excluded('unchanged_scope', kind):
                    pass
            tape.close()
            return 'original full worker result'
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output=Path(directory))
            with patch.object(candidate, 'ORIGINAL_WORKER', original_worker):
                result = candidate.worker(args, object(), 2048)
            self.assertEqual(result, 'original full worker result')
            self.assertIs(candidate.projection.Tape, original_tape)
            import json
            identity = json.loads((args.output/'lowlevel_capture_identity.json').read_text())
            self.assertEqual(identity['status'], 'complete')
            self.assertTrue(identity['sources_unchanged'])
            self.assertFalse(identity['asynchronous_queue_added'])
            self.assertFalse(identity['auxiliary_outputs_changed'])
            self.assertEqual(len(identity['capture_audits']), 1)
            self.assertEqual(identity['capture_audits'][0]['preparation']['calls'], 1)

    def test_worker_exception_restores_original_class(self):
        original_tape = candidate.projection.Tape
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output=Path(directory))
            with patch.object(candidate, 'ORIGINAL_WORKER', side_effect=RuntimeError('original failed')):
                with self.assertRaisesRegex(RuntimeError, 'original failed'):
                    candidate.worker(args, object(), 2048)
            self.assertIs(candidate.projection.Tape, original_tape)

    def test_original_boundaries_and_output_validation_not_copied_or_edited(self):
        source = Path(candidate.__file__).read_text(encoding='utf-8')
        self.assertIn('ORIGINAL_WORKER(args, guard, N)', source)
        self.assertNotIn('native.forward(', source)
        self.assertNotIn('optical_enabled=False', source)
        self.assertNotIn('make_graphed_callables(', source)
        self.assertNotIn('projection.__file__ =', source)

    def test_thin_runner_is_LG_only_and_freezes_original_and_candidate(self):
        # Linux entry imports fcntl, so test this small entry's own functions
        # with a fake original suite on Windows rather than importing Torch.
        runner = HERE/'run_lgvq_lowlevel_20261026.py'
        tree = ast.parse(runner.read_text(encoding='utf-8'))
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        called = []
        entry = SimpleNamespace(
            REPO=HERE.parents[3],
            base=SimpleNamespace(sha=lambda p: 'sha-'+Path(p).name),
            suite=lambda args: called.append(args),
        )
        original_entry = HERE/'run_corrected_boundaries_20261026.py'
        namespace = dict(
            argparse=argparse, json=json, os=os, Path=Path,
            entry=entry, projection=candidate.projection, candidate=candidate,
            PREVIOUS_SOURCES=lambda: {'retained_dependency.py': 'retained-sha'},
            ORIGINAL_ENTRY=original_entry, THIS_ENTRY=runner,
        )
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(runner), 'exec'), namespace)
        with patch.object(candidate.projection, 'worker'):
            namespace['main'](['--phase', 'suite', '--output', 'uncreated_run', '--deadline', '9999999999'])
        self.assertEqual(len(called), 1)
        self.assertTrue(called[0].lg_only)
        self.assertEqual(called[0].system, 'lg')
        self.assertEqual(called[0].videos, 2048)
        self.assertEqual(called[0].repeat, 1)
        self.assertEqual(entry.N, 2048)
        frozen = namespace['sources']()
        self.assertEqual(frozen['retained_dependency.py'], 'retained-sha')
        for path in (original_entry, runner, Path(candidate.__file__), Path(candidate.projection.__file__)):
            self.assertIn(path.relative_to(entry.REPO).as_posix(), frozen)


if __name__ == '__main__':
    unittest.main()
