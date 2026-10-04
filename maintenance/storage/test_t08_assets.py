from pathlib import Path
import tempfile
import unittest

from check_t08_assets import relative_image_content, audit_split


class T08Assets(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'image.jpg').write_bytes(b'synthetic image')
        self.row = {'sample_id': 'one', 'image_path': 'image.jpg',
                    'split': 'train', 'product_id': 'product'}

    def test_relative_content(self):
        actual = relative_image_content([self.row], self.root)
        self.assertEqual(actual['count'], 1)
        self.assertEqual(actual['split_counts'], {'train': 1})

    def test_relative_content_is_sensitive_to_bytes(self):
        first = relative_image_content([self.row], self.root)
        (self.root / 'image.jpg').write_bytes(b'changed bytes')
        self.assertNotEqual(first, relative_image_content([self.row], self.root))

    def test_traversal_and_absolute_rejected(self):
        for value in ['../image.jpg', str(self.root / 'image.jpg')]:
            with self.assertRaises(ValueError):
                relative_image_content([dict(self.row, image_path=value)], self.root)

    def test_split_contract(self):
        self.assertEqual(audit_split([self.row], [self.row], [],
                                    [{'product_id': 'product'}]),
                         {'train': 1, 'test': 0, 'titles': 1})

    def test_overlap_rejected(self):
        with self.assertRaises(ValueError):
            audit_split([self.row], [self.row], [self.row], [{'product_id': 'product'}])

    def test_changed_split_row_rejected(self):
        with self.assertRaises(ValueError):
            audit_split([self.row], [dict(self.row, product_id='other')], [],
                        [{'product_id': 'product'}])

    def test_wrong_title_library_rejected(self):
        with self.assertRaises(ValueError):
            audit_split([self.row], [self.row], [], [{'product_id': 'other'}])


if __name__ == '__main__':
    unittest.main()
