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

    def test_import_and_adaptation_do_not_import_torch(self):
        dev.load_candidate()
        self.assertNotIn('torch', sys.modules)

    def test_no_formal_writer_and_no_power_setter(self):
        source = Path(dev.__file__).read_text(encoding='utf8')
        self.assertIn('qualified_for_final_hybrid_table=False', source)
        self.assertNotIn('--power-limit', source)
        self.assertNotIn('allclose', source)
        self.assertIn("[('baseline', False), ('event_pool', True)]", source)


if __name__ == '__main__':unittest.main()
