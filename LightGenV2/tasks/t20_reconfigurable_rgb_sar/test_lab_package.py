import unittest
import numpy as np
from .build_lab_package import PINS, datasets_for, selected_training


class LabPackageTests(unittest.TestCase):
    def source(self):
        return dict(labels=np.tile(np.repeat(np.arange(10), 60), 2),
            domains=np.repeat([0, 1], 600), ids=np.asarray([f'train-{i}' for i in range(1200)]))

    def test_budget_and_deterministic_identity_selection(self):
        source = self.source()
        rows = selected_training(source)
        np.testing.assert_array_equal(rows, selected_training(source))
        self.assertEqual(len(set(rows)), 1000)
        for d in (0, 1):
            np.testing.assert_array_equal(np.bincount(source['labels'][rows][source['domains'][rows] == d]), [50]*10)
        # Performance and image values cannot influence the subset.
        changed = dict(source, predictions=np.zeros(1200), images=np.zeros((1200, 1)))
        np.testing.assert_array_equal(rows, selected_training(changed))

    def test_no_missing_class_replaced_from_test(self):
        source = self.source()
        source['labels'][source['labels'] == 9] = 8
        with self.assertRaises(ValueError):
            selected_training(source)
        with self.assertRaises(ValueError):
            selected_training(self.source(), 51)

    def test_duplicate_identity_rejected(self):
        source = self.source()
        source['ids'][0] = source['ids'][1]
        with self.assertRaises(ValueError):
            selected_training(source)

    def test_original_unequal_training_domain_protocol_is_explicit(self):
        domains = np.repeat([0, 1], 500)
        self.assertEqual(len(datasets_for('LG', domains)), 1000)
        self.assertEqual(datasets_for('D2NN', domains), list(range(500)))
        self.assertEqual((1000+2000)*3+(500+2000)*2, 14000)
        with self.assertRaises(ValueError):
            datasets_for('T21', domains)
        with self.assertRaises(ValueError):
            datasets_for('D2NN', domains[:999])

    def test_targets_use_one_checkpoint_per_case(self):
        self.assertEqual({d: n-PINS['LG']['max_drop_correct'] for d, n in PINS['LG']['correct'].items()}, {'RGB':819, 'SAR':680})
        self.assertEqual({d: n-PINS['D2NN']['max_drop_correct'] for d, n in PINS['D2NN']['correct'].items()}, {'RGB':748, 'SAR':541})
        self.assertTrue(PINS['LG']['sha256'].startswith('0e0587'))
        self.assertTrue(PINS['D2NN']['sha256'].startswith('14b36e'))


if __name__ == '__main__':
    unittest.main()
