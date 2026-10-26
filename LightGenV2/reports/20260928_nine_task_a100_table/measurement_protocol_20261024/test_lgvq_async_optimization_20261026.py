"""No model/CUDA: candidate policy, restoration and unchanged math calls."""
import gc
import ast
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from lgvq_async_optimization_20261026 import (
    prepare_without_cache_purge, runtime_gc_policy, detailed_boundaries)


class PolicyTests(unittest.TestCase):
    def test_prompt_arithmetic_relocated_not_reassociated(self):
        from lgvq_async_optimization_20261026 import separate_prompt_adapter
        from pathlib import Path
        path=Path(__file__).resolve().parents[4]/'LightGenV2/tasks/t06_video_quality_assessment/models/multivideo9x4.py'
        tree=ast.parse(path.read_text(encoding='utf-8'))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MultiVideo9x4OpticalVQA')
        forward=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
        import lgvq_scoreonly_20261026 as score
        import copy,hashlib,textwrap
        score_tree,_=score.score_forward_AST(ast.unparse(forward))
        helper=ast.parse((Path(__file__).parent/'lgvq_hoisted_pipeline_20261026.py').read_text(encoding='utf-8'))
        partition=next(n for n in helper.body if isinstance(n,ast.FunctionDef) and n.name=='partition_forward')
        ns=dict(ast=ast,copy=copy,hashlib=hashlib,textwrap=textwrap,
            PREFIX_NAMES=('batch','videos','vision','prompt_tokens','prompt_mask'))
        exec(compile(ast.Module(body=[partition],type_ignores=[]),'<actual_partition>','exec'),ns)
        original,_=ns['partition_forward'](ast.unparse(score_tree))
        changed,receipt=separate_prompt_adapter(original)
        self.assertEqual(ast.dump(original.body[1]),ast.dump(changed.body[1]))
        self.assertTrue(receipt['original_prompt_adapter_arithmetic_unchanged'])
        self.assertEqual(changed.body[-1].name,'prepare_text_adapter')
        self.assertEqual(len(changed.body[-1].body),8)
        self.assertNotIn('prompt_base = self.language_adapter',ast.unparse(changed.body[0]))
        self.assertIn('prompt_base = self.language_adapter',ast.unparse(changed.body[-1]))
    def test_single_synchronization_without_forced_gc_or_cache_purge(self):
        cuda=SimpleNamespace(synchronize=Mock(),empty_cache=Mock())
        tape=SimpleNamespace(preparation=None,preparation_calls=0,
            capture_audit={},torch=SimpleNamespace(cuda=cuda))
        prepare_without_cache_purge(tape);prepare_without_cache_purge(tape)
        cuda.synchronize.assert_called_once();cuda.empty_cache.assert_not_called()
        self.assertEqual(tape.preparation_calls,1)
        self.assertFalse(tape.preparation['forced_gc_collect'])
        self.assertEqual(tape.preparation['extra_forward_calls'],0)

    def test_gc_policy_restores_even_after_exception(self):
        before=gc.isenabled()
        with self.assertRaisesRegex(RuntimeError,'test'):
            with runtime_gc_policy(True):
                self.assertFalse(gc.isenabled());raise RuntimeError('test')
        self.assertEqual(gc.isenabled(),before)

    def test_detailed_scopes_call_original_once_and_restore(self):
        from contextlib import contextmanager
        seen=[]
        @contextmanager
        def scope(label,kind):
            seen.append((label,kind));yield
        a=Mock(return_value=3);b=Mock(return_value=4);restored=Mock()
        model=SimpleNamespace(frame_merger=SimpleNamespace(forward=a),readout=SimpleNamespace(forward=b))
        close,ledger=detailed_boundaries(model,SimpleNamespace(excluded=scope),lambda m,t:(restored,[]))
        self.assertEqual(model.frame_merger.forward(1),3)
        self.assertEqual(model.readout.forward(2),4)
        a.assert_called_once_with(1);b.assert_called_once_with(2)
        self.assertEqual(seen,[('frame_merger','electronics'),('readout','electronics')])
        self.assertEqual(len(ledger),2);close();restored.assert_called_once()
        self.assertIs(model.frame_merger.forward,a);self.assertIs(model.readout.forward,b)


if __name__=='__main__':unittest.main()
