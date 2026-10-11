import unittest
from prepare_caltech50 import split_groups


class SplitTests(unittest.TestCase):
    def test_duplicate_group_never_crosses_split_and_all_photos_retained(self):
        rows = [dict(label=k, original_hash=f'o{k}_{i}', cache_hash=f'c{k}_{i}')
                for k in range(2) for i in range(30)]
        rows.append(dict(rows[0]))
        split, groups = split_groups(rows)
        self.assertEqual(groups, 60)
        self.assertEqual(sum(map(len, split.values())), 61)
        owner = {i:k for k, values in split.items() for i in values}
        self.assertEqual(owner[0], owner[60])
        for k in range(2):
            self.assertEqual(len({rows[i]['original_hash'] for i in split['train'] if rows[i]['label']==k}), 20)
            self.assertEqual(len({rows[i]['original_hash'] for i in split['val'] if rows[i]['label']==k}), 5)
        self.assertEqual(split, split_groups(rows)[0])

    def test_conflicting_identical_image_labels_fail_before_split(self):
        with self.assertRaisesRegex(AssertionError, 'Cross-category'):
            split_groups([dict(label=0, original_hash='x', cache_hash='a'),
                          dict(label=1, original_hash='x', cache_hash='b')])


if __name__ == '__main__': unittest.main()
