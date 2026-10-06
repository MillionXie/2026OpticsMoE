import unittest
from maintenance.git_safety.check_t08_readout_cache import selected_indexes


class ReadoutSelectionTests(unittest.TestCase):
    def test_original_train_selection_is_fixed_and_balanced(self):
        rows = [{'label': str(i // 48)} for i in range(4800)]
        indexes = selected_indexes(rows, 'train')
        self.assertEqual(indexes, selected_indexes(rows, 'train'))
        self.assertEqual(len(set(indexes)), 800)
        self.assertEqual(indexes, sorted(indexes))
        for label in range(100): self.assertEqual(sum(i // 48 == label for i in indexes), 8)

    def test_short_train_rejected(self):
        with self.assertRaises(ValueError): selected_indexes([{'label': '0'}], 'train')

    def test_original_test_order(self):
        self.assertEqual(selected_indexes([{}] * 2400, 'test'), list(range(2400)))
        with self.assertRaises(ValueError): selected_indexes([{}], 'test')
