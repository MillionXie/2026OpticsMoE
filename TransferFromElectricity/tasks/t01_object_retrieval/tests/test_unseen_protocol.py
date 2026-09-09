import unittest
from types import SimpleNamespace
from TransferFromElectricity.tasks.t01_object_retrieval.unseen_protocol import validate_classes, select_support


class UnseenProtocolTests(unittest.TestCase):
    def test_rejects_source_or_cross_task_class_overlap(self):
        validate_classes(list(range(10)), {'a':list(range(10,20)), 'b':list(range(20,30))})
        for novel in ({'a':list(range(9,19))}, {'a':list(range(10,20)), 'b':list(range(19,29))}):
            with self.assertRaises(ValueError): validate_classes(list(range(10)), novel)

    def test_nested_balanced_reproducible_support(self):
        rows = [SimpleNamespace(sample_id=f'{label}:{i}',sku_index=label,sku_id=label+10,source_split='official_train')
                for label in range(10) for i in range(50)]
        five = select_support(rows,5,101); twenty = select_support(list(reversed(rows)),20,101)
        self.assertEqual(len(five),50)
        self.assertEqual(len(twenty),200)
        self.assertTrue({s.sample_id for s in five} <= {s.sample_id for s in twenty})
        self.assertEqual([s.sample_id for s in five], [s.sample_id for s in select_support(rows,5,101)])
        self.assertNotEqual([s.sample_id for s in five], [s.sample_id for s in select_support(rows,5,202)])
        rows[0].source_split = 'official_test'
        with self.assertRaises(ValueError): select_support(rows,5,101)


if __name__ == '__main__': unittest.main()
