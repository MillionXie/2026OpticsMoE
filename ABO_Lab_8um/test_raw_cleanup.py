import json
from pathlib import Path
import tempfile
import unittest
from common import sha, write
from raw_cleanup import validate_record_files


class CleanupTests(unittest.TestCase):
    def test_missing_tiff_needs_exact_audit_and_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); d = root/'ccd/image_a'; d.mkdir(parents=True)
            png = d/'vision_global.png'; png.write_bytes(b'kept pixel bytes')
            rp = d/'vision_global.record.json'
            meta = dict(sample='image_a',stage='vision_global',files={'vision_global.png':sha(png),'vision_global.tif':'original_hash'})
            write(rp,meta)
            with self.assertRaises(FileNotFoundError): validate_record_files(root,d,rp,meta)
            row = dict(action='deleted',sample='image_a',stage='vision_global',path='ccd/image_a/vision_global.tif',
                record_sha256=sha(rp),recorded_tiff_sha256='original_hash',png_sha256=sha(png))
            (root/'raw_tiff_cleanup.jsonl').write_text(json.dumps(row)+'\n',encoding='utf-8')
            validate_record_files(root,d,rp,meta)
            png.write_bytes(b'changed')
            with self.assertRaises(ValueError): validate_record_files(root,d,rp,meta)


if __name__ == '__main__': unittest.main()
