"""Synthetic personal asset tests; never access original photos or annotations."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from check_t02_assets import personal_content


class PersonalAssetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'photo.jpg').write_bytes(b'fake photo')
        self.row = {'id': 'photo_0', 'image': 'photo.jpg',
                    'image_sha256': hashlib.sha256(b'fake photo').hexdigest()}

    def test_verified_content(self):
        result = personal_content(self.root, {'images': [self.row]})
        self.assertEqual(result['image_count'], 1)
        self.assertEqual(len(result['image_content_manifest_sha256']), 64)

    def test_changed_image_rejected(self):
        (self.root / 'photo.jpg').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            personal_content(self.root, {'images': [self.row]})

    def test_duplicate_identity_rejected(self):
        with self.assertRaises(ValueError):
            personal_content(self.root, {'images': [self.row, self.row]})

    def test_traversal_rejected(self):
        with self.assertRaises(ValueError):
            personal_content(self.root, {'images': [dict(self.row, image='../photo.jpg')]})

    def test_absolute_path_rejected(self):
        with self.assertRaises(ValueError):
            personal_content(self.root, {'images': [dict(self.row, image=str(self.root / 'photo.jpg'))]})


if __name__ == '__main__':
    unittest.main()
