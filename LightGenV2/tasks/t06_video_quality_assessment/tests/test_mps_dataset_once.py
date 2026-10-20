"""CPU-only contract tests; never execute or prewarm a model."""
import ast
import inspect
import unittest
from LightGenV2.tasks.t06_video_quality_assessment import measure_mps_dataset_once as m


class DatasetOnce(unittest.TestCase):
    def test_include_gap_but_exclude_every_next_preprocessing(self):
        calls=[dict(preprocess_start=0,model_call_start=1,first_block_start=2,gpu_output=3),
               dict(preprocess_start=6,model_call_start=8,first_block_start=9,gpu_output=11)]
        intervals,compute,gap=m.charged_intervals(calls)
        self.assertEqual(intervals,[(2.,3.),(3.,6.),(9.,11.)])
        self.assertEqual((compute,gap),(3.,3.))
        self.assertEqual(sum(b-a for a,b in intervals),6.)
        # Continuous first-TF->last-output9s, minus next-preprocessing3s.
        self.assertEqual(11-2-(9-6),compute+gap)

    def test_first_and_terminal_handling_not_charged(self):
        intervals,compute,gap=m.charged_intervals([
            dict(preprocess_start=0,model_call_start=2,first_block_start=3,gpu_output=4)])
        self.assertEqual(intervals,[(3.,4.)]);self.assertEqual((compute,gap),(1.,0.))

    def test_cost_boundary_has_199_gaps_not200(self):
        calls=[dict(preprocess_start=i*10,model_call_start=i*10+1,
                    first_block_start=i*10+2,gpu_output=i*10+3) for i in range(201)]
        _,compute,gap=m.charged_intervals(calls[:200])
        self.assertEqual(compute,200);self.assertEqual(gap,199*7)

    def test_overlapping_or_missing_intervals_fail(self):
        with self.assertRaises(ValueError):m.charged_intervals([])
        with self.assertRaises(ValueError):m.charged_intervals([
            dict(preprocess_start=0,model_call_start=1,first_block_start=2,gpu_output=4),
            dict(preprocess_start=3,model_call_start=5,first_block_start=6,gpu_output=7)])

    def test_finite_original_scope(self):
        c=m.config()
        self.assertEqual(c['batch'],1);self.assertEqual(c['explicit_warmup_forwards'],0)
        self.assertEqual(len(c['models'])*len(c['resource_caps'])*c['repeats'],24)
        self.assertNotIn('deepseek',c['models']);self.assertNotIn('kimi',c['models'])
        self.assertTrue(c['historical_Qwen_B1_unchanged'])

    def test_no_forward_in_warmup_or_runtime_setup(self):
        tree=ast.parse(inspect.getsource(m.runtime))
        outer_calls=[]
        class SetupCalls(ast.NodeVisitor):
            def visit_FunctionDef(self,node):pass
            def visit_Call(self,node):
                if isinstance(node.func,ast.Name):outer_calls.append(node.func.id)
                self.generic_visit(node)
        for statement in tree.body[0].body:SetupCalls().visit(statement)
        self.assertNotIn('forward',outer_calls);self.assertNotIn('compute',outer_calls)
        worker=ast.parse(inspect.getsource(m.worker))
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
                             and n.func.id in ('warm','warmup') for n in ast.walk(worker)))

    def test_rental_profile_has_same_boundary_but_separate_run(self):
        import json
        c=json.loads((m.TASK/'configs/baselines/mps_dataset_once_rental_20261021.json').read_text())
        original=m.config()
        for field in ('batch','test_videos','cost_videos','repeats','timing_protocol',
                      'included','excluded','explicit_warmup_forwards','host_proxy_w'):
            self.assertEqual(c[field],original[field])
        self.assertEqual(c['models'],['deepseek','kimi'])
        self.assertEqual(c['power_limit_w'],400)
        self.assertNotEqual(c['gpu_uuid'],original['gpu_uuid'])


if __name__=='__main__':unittest.main()
