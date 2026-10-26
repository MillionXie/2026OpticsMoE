import unittest
from rental_energy_triples_20261026 import retained_intervals


class AccountingTests(unittest.TestCase):
    def row(self, a, b, c, d, tokens=()):
        return dict(CPU_start=a, CPU_end=b, H2D_end=c, GPU_ready_host=d,
            tokenizer_intervals=[dict(start=x, end=y) for x, y in tokens])

    def test_keep_tokenizer_and_all_intervideo_waits(self):
        rows = [self.row(0, 2, 3, 5, [(1, 1.5)]), self.row(6, 8, 9, 11, [(7, 7.5)])]
        self.assertEqual(retained_intervals(rows), [(1, 1.5), (3, 6), (7, 7.5), (9, 11)])

    def test_nested_tokenizer_not_double_charged(self):
        self.assertEqual(retained_intervals([self.row(0, 3, 4, 6, [(1, 2), (1.2, 1.6), (1.9, 2.5)])]),
            [(1, 2.5), (4, 6)])

    def test_reject_outside_tokenizer(self):
        with self.assertRaises(ValueError):
            retained_intervals([self.row(0, 1, 2, 3, [(1, 1.5)])])

    def test_no_tokenizer(self):
        self.assertEqual(retained_intervals([self.row(0, 1, 2, 3)]), [(2, 3)])

    def test_overlapping_rows_rejected(self):
        with self.assertRaises(ValueError):
            retained_intervals([self.row(0, 2, 3, 5), self.row(2, 4, 5, 7)])


if __name__ == '__main__':
    unittest.main()
