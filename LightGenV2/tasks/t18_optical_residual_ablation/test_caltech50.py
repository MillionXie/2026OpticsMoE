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

    def test_expansion_keeps_previous_train_val_out_of_new_test(self):
        rows = [dict(name=f'img{i}', label=0, original_hash=f'o{i}', cache_hash=f'c{i}') for i in range(90)]
        previous = {f'img{i}':'train' for i in range(20)}
        previous.update({f'img{i}':'val' for i in range(20,25)})
        split, _ = split_groups(rows, 20261012, 60, 10, previous)
        self.assertEqual([len(split[k]) for k in ['train','val','test']], [60,10,20])
        roles = {rows[i]['name']:k for k, indices in split.items() for i in indices}
        self.assertTrue(all(roles[name] == role for name, role in previous.items()))
        self.assertFalse({rows[i]['name'] for i in split['test']} & set(previous))


if __name__ == '__main__': unittest.main()
