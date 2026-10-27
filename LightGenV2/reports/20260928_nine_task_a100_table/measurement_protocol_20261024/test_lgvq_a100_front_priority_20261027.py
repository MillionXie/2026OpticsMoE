import ast
import inspect
import sys
from types import SimpleNamespace
import unittest
import run_lgvq_a100_front_priority_20261027 as candidate


class PriorityTests(unittest.TestCase):
    def test_only_frontend_creation_and_actual_priority_identity_added(self):
        source=inspect.getsource(candidate.cpp.native.worker)
        before=candidate.WORKER_AST(source);tree=candidate.worker_AST(source)
        class Reverse(ast.NodeTransformer):
            def visit_Assign(self,n):
                if isinstance(n.value,ast.Tuple) and ast.unparse(n.targets[0])=='(fs, bs)':n.value.elts[0].keywords=[]
                return n
            def visit_Expr(self,n):
                if 'a100_front_priority_identity.json' in ast.unparse(n):return None
                return n
        self.assertEqual(ast.dump(Reverse().visit(tree)),ast.dump(before))

    def test_closure_and_global_restoration_no_torch(self):
        self.assertNotIn('torch',sys.modules)
        baseline=candidate.cpp.core.ORIGINAL_SOURCES();before=candidate.cpp.ENTRY
        with candidate.configured():
            result=candidate.source_map()
            self.assertEqual({k:result[k] for k in baseline},baseline)
            self.assertEqual(len(result),len(baseline)+7)
        self.assertEqual(candidate.cpp.ENTRY,before)

    def test_only_auxiliary_creation_scoped_and_actual_priority_saved(self):
        self.assertNotIn('torch',sys.modules)
        fake=SimpleNamespace(cuda=SimpleNamespace(Stream=lambda **kw:SimpleNamespace(priority=kw.get('priority',0))))
        saved_install=candidate.INSTALL;original=fake.cuda.Stream
        def install(front,partitions):fake.cuda.Stream(device=0);return dict(partitions=partitions)
        sys.modules['torch']=fake;candidate.INSTALL=install
        try:
            info=candidate.install(object())
            self.assertEqual(info['auxiliary_stream_priorities'],[-1])
            self.assertIs(fake.cuda.Stream,original)
        finally:candidate.INSTALL=saved_install;del sys.modules['torch']


if __name__=='__main__':unittest.main()
