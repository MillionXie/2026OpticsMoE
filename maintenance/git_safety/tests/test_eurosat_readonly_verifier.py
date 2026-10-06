import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from LightGenV2.demo_check.verify_eurosat_image_archive import verify


class ArchiveTests(unittest.TestCase):
    def fixture(self, root, pixel_hash=None):
        from PIL import Image
        image = Image.new('RGB', (56, 56), (1, 2, 3))
        buffer = io.BytesIO(); image.save(buffer, format='PNG'); raw = buffer.getvalue()
        split = {'records': [{'path': 'images/sample.png', 'domain': 'A', 'split': 'train'}]}
        manifest = {'images/sample.png': {'sha256': hashlib.sha256(raw).hexdigest(), 'pixel_sha256': pixel_hash or hashlib.sha256(image.tobytes()).hexdigest()}}
        (root/'split.json').write_text(json.dumps(split))
        (root/'manifest.json').write_text(json.dumps(manifest))
        with zipfile.ZipFile(root/'data.zip', 'w') as package:
            package.writestr('images/sample.png', raw)
            package.writestr('SPLIT.json', json.dumps(split))
            package.writestr('IMAGE_MANIFEST.json', json.dumps(manifest))
        archive = (root/'data.zip').read_bytes()
        (root/'report.json').write_text(json.dumps({'passed': True, 'state': 'complete', 'archive_bytes': len(archive), 'archive_sha256': hashlib.sha256(archive).hexdigest(), 'image_manifest_sha256': hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()}))
        return [root/'data.zip', root/'report.json', root/'split.json', root/'manifest.json']

    def test_validation_preserves_all_input_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); args = self.fixture(root)
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            self.assertTrue(verify(*args, expected_images=1)['read_only'])
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})

    def test_wrong_pixels_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp), '0'*64)
            with self.assertRaisesRegex(ValueError, 'pixel contract'):
                verify(*args, expected_images=1)


if __name__ == '__main__':
    unittest.main()
