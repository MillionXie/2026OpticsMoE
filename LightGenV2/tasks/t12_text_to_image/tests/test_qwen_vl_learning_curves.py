import json
import tempfile
import unittest
from pathlib import Path

from LightGenV2.tasks.t12_text_to_image.reports.qwen_vl_data_review_20261027.learning_curves import current_snapshot


class SnapshotTests(unittest.TestCase):
    def test_rejects_in_progress_mixed_epoch_files(self):
        with tempfile.TemporaryDirectory() as name:
            run = Path(name)
            data = {'count': 96, 'mean_per_image_psnr_db': 22., 'mean_lpips': .25,
                    'mean_object_crop_lpips': .33, 'by_mode': {}, 'lpips_by_mode': {},
                    'object_crop_lpips_by_mode': {}, 'by_category_mode': {}}
            (run/'val_latest.json').write_text(json.dumps(data))
            (run/'train_latest.json').write_text(json.dumps({**data, 'count': 12, 'mean_per_image_psnr_db': 27.}))
            h = [{'step': 600, 'epoch': 5, 'val_psnr': 22., 'val_lpips': .25, 'train_psnr': 27.}]
            self.assertEqual(current_snapshot(run, h)['train_diagnostic']['count'], 12)
            (run/'val_latest.json').write_text(json.dumps({**data, 'mean_lpips': .24}))
            self.assertIsNone(current_snapshot(run, h))
            self.assertIsNone(current_snapshot(run, []))


if __name__ == '__main__':
    unittest.main()
