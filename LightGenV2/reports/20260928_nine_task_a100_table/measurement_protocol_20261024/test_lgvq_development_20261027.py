"""Fail-closed hardware portability tests; no CUDA/model execution."""
import ast
from pathlib import Path
import sys
import unittest

import run_lgvq_development_20261027 as dev


class DevelopmentTests(unittest.TestCase):
    def test_exact_hardware_changes_and_no_algorithm_edits(self):
        source = dev.WORKER.read_text(encoding='utf8')
        tree, counts = dev.development_adapter(source)
        self.assertEqual(counts, [1, 1, 1, 1])
        expected = source.replace("GPU = 'GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55'", f'GPU = {dev.GPU!r}')
        expected = expected.replace('multi_processor_count != 108', 'multi_processor_count != 128')
        expected = expected.replace('power_limit != 250', 'power_limit != 450.0')
        expected = expected.replace('actual_visible_multiprocessors=108', 'actual_visible_multiprocessors=128')
        self.assertEqual(ast.dump(tree, include_attributes=False), ast.dump(ast.parse(expected), include_attributes=False))

    def test_drift_or_duplicate_rejected(self):
        source = dev.WORKER.read_text(encoding='utf8')
        for changed in [source.replace('power_limit != 250', 'power_limit != 251'),
                        source+'\nGPU = "GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55"\n']:
            with self.assertRaises(ValueError):dev.development_adapter(changed)

    def test_original_numeric_reference_remains(self):
        candidate, _ = dev.load_candidate()
        self.assertEqual(candidate.REFERENCE_SHA256, '6fc71216c3deaa4975532256c3d50367203143e352268372da39ad89ea216a4e')
        import numpy as np
        array = np.zeros((128, 16), dtype=np.float32)
        candidate.validate_predictions(array, array.copy())
        changed = array.copy(); changed[-1, -1] = -0.
        with self.assertRaises(ValueError):candidate.validate_predictions(changed, array)

    def test_same_card_reference_is_exact_and_separate(self):
        self.assertEqual(dev.SAME_CARD_REFERENCE.name, 'baseline')
        self.assertEqual(dev.SAME_CARD_REFERENCE.parent.name, 'lgvq_rtx4090_development_20261027_01')
        self.assertEqual(dev.SAME_CARD_SHA256, '31fedd4d30d24b705774007be0baa34f78c34a7bfa21b8890718a7334fd3bc88')
        source = Path(dev.__file__).read_text(encoding='utf8')
        self.assertIn("[('event_pool', True)] if reference", source)
        self.assertIn('A100_bitwise_equivalence_claimed=False', source)

    def test_import_and_adaptation_do_not_import_torch(self):
        dev.load_candidate()
        self.assertNotIn('torch', sys.modules)

    def test_no_formal_writer_and_no_power_setter(self):
        source = Path(dev.__file__).read_text(encoding='utf8')
        self.assertIn('qualified_for_final_hybrid_table=False', source)
        self.assertNotIn('--power-limit', source)
        self.assertNotIn('allclose', source)
        self.assertIn("[('baseline', False), ('event_pool', True)]", source)

    def test_CPU_thread_candidate_is_isolated(self):
        source = Path(dev.__file__).read_text(encoding='utf8')
        self.assertIn('choices=[1,8], default=8',source)
        self.assertIn('torch.set_num_threads(args.cpu_threads)',source)
        self.assertIn('not args.same_card_reference or args.share_graph_pools or args.reuse_events',source)
        self.assertIn("[('cpu_threads1', False)]",source)

    def test_text_stream_candidate_is_isolated_and_forwarded(self):
        source=Path(dev.__file__).read_text(encoding='utf8')
        self.assertIn("[('parallel_text', False)]",source)
        self.assertIn("if args.parallel_text:command.append('--parallel-text')",source)
        self.assertIn("args.parallel_text != (args.output.name=='parallel_text')",source)
        self.assertIn('or args.reuse_events or args.once_prompt_render or args.channels_last_3d',source)
        self.assertNotIn('args.parallel_text = False',source)

    def test_text_stream_fork_and_join_remain_inside_original_prefix(self):
        import inspect
        from lgvq_async_optimization_20261026 import parallel_front
        source=inspect.getsource(parallel_front)
        self.assertLess(source.index('text_stream.wait_stream'),source.index("front.embedding(raw['ids'])"))
        self.assertLess(source.index('prepared=text_fn'),source.index('ready.record()'))
        self.assertLess(source.index("front.visual.patch_embed(raw['pixels'])"),source.index('wait_event(ready)'))
        self.assertLess(source.index('wait_event(ready)'),source.index('return dict'))
        self.assertEqual(source.count('front.embedding('),1)
        self.assertEqual(source.count('text_fn(model,'),1)
        self.assertNotIn('synchronize()',source)

    def test_patch_linear_candidate_is_isolated(self):
        source=Path(dev.__file__).read_text(encoding='utf8')
        self.assertIn("[('patch_linear', False)]",source)
        self.assertIn("if args.patch_linear:command.append('--patch-linear')",source)
        self.assertIn("args.patch_linear != (args.output.name=='patch_linear')",source)
        self.assertIn('or args.channels_last_3d or args.parallel_text',source)


if __name__ == '__main__':unittest.main()
