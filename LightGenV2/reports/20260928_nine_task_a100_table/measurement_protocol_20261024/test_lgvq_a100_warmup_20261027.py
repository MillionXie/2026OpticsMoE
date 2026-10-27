import ast
import inspect
import sys
import unittest
import lgvq_a100_warmup_20261027 as warm
import run_lgvq_a100_warmup_20261027 as entry


class WarmupTests(unittest.TestCase):
    def test_exact_extra_forward_and_post_window_save_AST_only(self):
        source=inspect.getsource(entry.native.worker);tree=warm.worker_AST(source)
        self.assertEqual(sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_prewarm' for n in ast.walk(tree)),1)
        class Reverse(ast.NodeTransformer):
            def visit_ImportFrom(self,n):
                if n.module=='lgvq_a100_warmup_20261027':n.module='lgvq_async_raw_adapter_20261026'
                return n
            def visit_Assign(self,n):
                if isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='_prewarm':return None
                return self.generic_visit(n)
            def visit_Expr(self,n):
                if isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='_save_warm':return None
                return self.generic_visit(n)
            def visit_Call(self,n):
                n.keywords=[k for k in n.keywords if k.arg not in ('timing_profile','extra_warmup','additional_pre_window_real_fields')]
                for key in n.keywords:
                    if key.arg=='explicit_warmup_forwards':key.value=ast.Constant(0)
                return self.generic_visit(n)
        self.assertEqual(ast.dump(Reverse().visit(tree)),ast.dump(ast.parse(source)))

    def test_charge_warmup_to_measured_workload_not_diluted_by_extra_outputs(self):
        raw=dict(timing_profile=warm.PROFILE,explicit_warmup_forwards=1,host_start=3.,host_end=5.,
            extra_warmup=dict(profile=warm.PROFILE,extra_real_fields=1,extra_video_calls=16,
                actual_full_coherent_optical_passes=6,warm_scores_bitwise_equal_reference=True,
                neural_features_or_predictions_reused=False,host_start=1.,host_end=2.))
        info=warm.accounting(raw)
        self.assertEqual(info['inclusive_software_ms'],4000.)
        self.assertEqual(info['inclusive_software_ms_charged_per_measured_16'],4000/128)
        self.assertEqual(info['total_executed_video_calls'],2064)
        with self.assertRaises(ValueError):warm.accounting(dict(raw,explicit_warmup_forwards=2))

    def test_scope_is_not_accepted_as_original_cold_measurement(self):
        self.assertFalse(entry.accepted_report(dict(status='complete',qualified_for_final_hybrid_table=True)))
        self.assertNotIn('torch',sys.modules)
        original=entry.core.ORIGINAL_SOURCES();before=entry.core.original.accepted_report
        with entry.configured():
            frozen=entry.source_map()
            self.assertEqual({k:frozen[k] for k in original},original)
            self.assertEqual(len(frozen),len(original)+4)
        self.assertIs(entry.core.original.accepted_report,before)

    def test_one_complete_prefix_and_tail_extra_call_no_retry_loop(self):
        tree=ast.parse(inspect.getsource(warm.prewarm))
        names=[n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
        self.assertEqual(names.count('prefix'),1)
        self.assertEqual(names.count('tail'),1)
        self.assertFalse(any(isinstance(n,(ast.For,ast.While)) for n in ast.walk(tree)))


if __name__=='__main__':unittest.main()
