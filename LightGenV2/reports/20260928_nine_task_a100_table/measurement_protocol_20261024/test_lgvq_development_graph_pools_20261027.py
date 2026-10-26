import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import lgvq_development_graph_pools_20261027 as pools


class FakeTape:
    def __init__(self, torch, stream):self.torch=torch; self.stream=stream
    def start(self, kind, label):
        graph = self.torch.cuda.CUDAGraph()
        graph.capture_begin(capture_error_mode='global')
        return graph


class PoolTests(unittest.TestCase):
    def test_only_pool_hint_changes(self):
        import inspect,textwrap
        original=textwrap.dedent(inspect.getsource(FakeTape.start))
        transformed=pools.pool_hint_AST(original)
        expected=original.replace("capture_begin(capture_error_mode='global')",
            "capture_begin(pool=self.pool_for_current_stream(), capture_error_mode='global')")
        self.assertEqual(ast.dump(transformed),ast.dump(ast.parse(expected)))

    def test_fail_closed_on_drift_or_two_capture_calls(self):
        with self.assertRaises(ValueError):pools.pool_hint_AST('def start(self):\n    return None')
        with self.assertRaises(ValueError):pools.pool_hint_AST("def start(self):\n    graph.capture_begin(capture_error_mode='global')\n    graph.capture_begin(capture_error_mode='global')")

    def test_slots_and_streams_never_share(self):
        handles=[];calls=[]
        def handle():
            value=object(); handles.append(value); return value
        class Graph:
            def capture_begin(self,**kw):calls.append(kw)
        torch=SimpleNamespace(cuda=SimpleNamespace(graph_pool_handle=handle,CUDAGraph=Graph))
        cls=pools.tape_class(FakeTape)
        a=cls(torch,SimpleNamespace(cuda_stream=10)); b=cls(torch,SimpleNamespace(cuda_stream=10))
        a.start('electronics','first'); a.start('electronics','next'); b.start('electronics','other_slot')
        self.assertIs(calls[0]['pool'],calls[1]['pool'])
        self.assertIsNot(calls[0]['pool'],calls[2]['pool'])
        a.stream=SimpleNamespace(cuda_stream=20); a.start('electronics','body')
        self.assertIsNot(calls[0]['pool'],calls[3]['pool'])
        self.assertEqual(len(handles),3)
        self.assertTrue(all(c['capture_error_mode']=='global' for c in calls))


if __name__=='__main__':unittest.main()
