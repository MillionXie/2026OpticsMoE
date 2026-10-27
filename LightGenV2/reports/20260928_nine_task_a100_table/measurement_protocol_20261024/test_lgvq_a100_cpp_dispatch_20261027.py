import ast
import inspect
import sys
import unittest
import lgvq_a100_cpp_dispatch_20261027 as dispatch
import run_lgvq_a100_cpp_dispatch_20261027 as entry


class CPPTests(unittest.TestCase):
    def test_worker_only_replay_import_and_post_window_identity_save_change(self):
        source=inspect.getsource(entry.native.worker);tree=dispatch.worker_AST(source)
        class Reverse(ast.NodeTransformer):
            def visit_Import(self,n):
                if n.names[0].name=='lgvq_a100_cpp_dispatch_20261027':n.names[0].name='lgvq_async_replay_20261026'
                return n
            def visit_Expr(self,n):
                if 'a100_cpp_dispatch_identity.json' in ast.unparse(n):return None
                return n
        self.assertEqual(ast.dump(Reverse().visit(tree)),ast.dump(ast.parse(source)))

    def test_replay_only_body_host_loop_and_metadata_change(self):
        before=ast.parse(inspect.getsource(dispatch.original.replay_fields));tree=dispatch.replay_AST()
        loop=next(n for n in ast.walk(before) if isinstance(n,ast.For) and ast.unparse(n.target)=='segment')
        class Reverse(ast.NodeTransformer):
            def visit_Expr(self,n):
                if ast.unparse(n).startswith('observed.extend(_submit('):return loop
                return n
            def visit_Return(self,n):
                if isinstance(n.value,ast.Call):n.value.keywords=[k for k in n.value.keywords if k.arg!='native_dispatch']
                return n
        self.assertEqual(ast.dump(Reverse().visit(tree)),ast.dump(before))

    def test_original_closure_restore_no_torch(self):
        self.assertNotIn('torch',sys.modules)
        baseline=entry.core.ORIGINAL_SOURCES();before=entry.core.ENTRY
        with entry.configured():
            result=entry.source_map()
            self.assertEqual({k:result[k] for k in baseline},baseline)
            self.assertEqual(len(result),len(baseline)+6)
        self.assertEqual(entry.core.ENTRY,before)

    def test_CPP_retains_native_replay_and_actual_event_records(self):
        source=entry.ENTRY.with_name('lgvq_a100_cpp_dispatch_20261027.cpp').read_text(encoding='utf8')
        self.assertEqual(source.count('graphs[i]->replay();'),1)
        self.assertEqual(source.count('cudaEventRecord('),2)
        self.assertNotIn('capture_begin',source)
        self.assertNotIn('graph_exec_',source)
        self.assertIn('std::chrono::steady_clock::now()',source)


if __name__=='__main__':unittest.main()
