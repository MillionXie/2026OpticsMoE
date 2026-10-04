"""Synthetic fixtures only; no formal experiment assets are modified."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from check_t01_assets import file_identity, image_content


class ContentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'image.jpg').write_bytes(b'fixture image bytes')
        self.recorded = Path('/recorded/Caltech101')
        self.row = {'sample_id': 'a', 'split': 'train',
                    'image_path': str(self.recorded / 'image.jpg')}

    def test_file_bytes(self):
        actual = file_identity(self.root / 'image.jpg')
        self.assertEqual(actual, {'bytes': 19, 'sha256': hashlib.sha256(b'fixture image bytes').hexdigest()})

    def test_canonical_record(self):
        expected = {'sample_id': 'a', 'split': 'train', 'relative_path': 'image.jpg',
                    **file_identity(self.root / 'image.jpg')}
        payload = json.dumps(expected, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n'
        actual = image_content([self.row], self.root, self.recorded)
        self.assertEqual(actual['content_manifest_sha256'], hashlib.sha256(payload.encode()).hexdigest())
        self.assertEqual(actual['split_counts'], {'train': 1})

    def test_duplicate_identity_rejected(self):
        with self.assertRaises(ValueError):
            image_content([self.row, self.row], self.root, self.recorded)

    def test_duplicate_path_rejected(self):
        with self.assertRaises(ValueError):
            image_content([self.row, dict(self.row, sample_id='b')], self.root, self.recorded)

    def test_outside_recorded_root_rejected(self):
        with self.assertRaises(ValueError):
            image_content([dict(self.row, image_path='/elsewhere/image.jpg')], self.root, self.recorded)

    def test_traversal_rejected(self):
        with self.assertRaises(ValueError):
            image_content([dict(self.row, image_path=str(self.recorded / '../image.jpg'))], self.root, self.recorded)


if __name__ == '__main__':
    unittest.main()
