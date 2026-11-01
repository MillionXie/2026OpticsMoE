"""CPU-only mean-stage formula and native AST contracts; no model/GPU."""
import ast
import unittest
from pathlib import Path
import lgvq_mean_pipeline_20261101 as mean
import lgvq_front_scopes_20261101 as front


class MeanContract(unittest.TestCase):
    def fixture(self, cpu=.040):
        phases = ['emb_text','emb_conv3d','emb_position','adapter_visual_pool','adapter_visual',
                  'merger','adapter_language','decoder']+[f'oeo{i}' for i in range(1,7)]
        report = dict(raw_phases=[dict(field=f,phase=p,kernel_union_s=.001) for f in range(32) for p in phases]
                      +[dict(field=f,phase=f'optics{i}',kernel_union_s=99.) for f in range(32) for i in range(1,7)])
        rows = [dict(field=f,slot=s,decode_cpu_s=.010,RGB_cpu_s=.002,crop_cpu_s=.00001,
                     resize_cpu_s=cpu/3,normalization_cpu_s=cpu/3,patch_cpu_s=cpu/3)
                for f in range(32) for s in range(16)]
        return report, rows, [dict(thread_cpu_s=.001) for _ in range(32)]

    def test_balanced_mean_and_mean_tail_close(self):
        result = mean.estimate(*self.fixture(), 0)
        self.assertAlmostEqual(result['CPU_mean_s_per_field'], .040)
        self.assertTrue(result['simple_P_plus_Q_over_N_verified'])
        self.assertAlmostEqual(result['time16_s'], .040+result['mean_backend_after_image_s']/32)
        self.assertFalse(result['load_imbalance_charged'])

    def test_four_cumulative_scopes(self):
        values = [mean.estimate(*self.fixture(), v)['CPU_mean_s_per_field'] for v in range(4)]
        for got, expected in zip(values, [.040, .050, .052, .05201]):
            self.assertAlmostEqual(got, expected)

    def test_cannot_hide_backend_bottleneck(self):
        result = mean.estimate(*self.fixture(cpu=.00001), 0)
        self.assertFalse(result['simple_P_plus_Q_over_N_verified'])
        self.assertGreater(result['time16_s'], result['CPU_mean_s_per_field']+result['mean_backend_after_image_s']/32)

    def test_text_work_recorded_and_fixed_optics(self):
        result = mean.estimate(*self.fixture(), 0)
        self.assertAlmostEqual(result['backend_phase_mean_s_per_field']['tokenizer'], .001)
        self.assertAlmostEqual(result['backend_phase_mean_s_per_field']['optics1'], 1/1886+.000300+.0000307)

    def test_native_decoder_instrumentation_no_import(self):
        repo = Path(__file__).resolve().parents[4]
        source = (repo/'experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/cache_qwen_front.py').read_text(encoding='utf8')
        tree = ast.parse(source)
        fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'decode_frames')
        from types import SimpleNamespace
        ns = dict(Path=Path, Image=SimpleNamespace(Image=object), _sample_positions=lambda *a: [])
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<native_decoder_test>','exec'),ns)
        # inspect.getsource needs a real file-backed function, supplied by mock.
        from unittest.mock import patch
        with patch.object(front.inspect, 'getsource', return_value=ast.unparse(fn)):
            _, receipt = front.observed_decoder(ns['decode_frames'])
        self.assertEqual(receipt['instrumented_statements']['decode'], 7)
        self.assertTrue(receipt['crop_is_native_slice_no_extra_copy'])


if __name__ == '__main__':
    unittest.main()
