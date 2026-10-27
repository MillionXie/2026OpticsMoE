import unittest

from ..qwen_vl_instruction_probe import prompt_variants, select_train_sources


class InstructionProbeTest(unittest.TestCase):
    def test_only_color_changes_in_opposite_instruction(self):
        for category in ('lamp', 'pillow'):
            variants = prompt_variants(category)
            self.assertEqual(variants['short_red'].replace('red', 'blue'), variants['short_blue'])
            self.assertLess(len(variants['short_red'].split()), len(variants['long_red'].split()))

    def test_sources_are_train_only_even_when_dev_comes_first(self):
        rows = [{'split': 'val', 'category': 'lamp'}, {'split': 'train', 'category': 'lamp'},
                {'split': 'test', 'category': 'pillow'}, {'split': 'train', 'category': 'pillow'}]
        self.assertEqual(select_train_sources(rows), [1, 3])
        with self.assertRaises(StopIteration):
            select_train_sources(rows[:3])


if __name__ == '__main__':
    unittest.main()
