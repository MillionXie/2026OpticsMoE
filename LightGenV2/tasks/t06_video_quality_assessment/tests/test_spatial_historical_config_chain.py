"""Historical release closure checks; no Torch, data, checkpoint or device use."""
import hashlib
import json
from pathlib import Path
import unittest

from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.settings import load_settings

ROOT = Path(__file__).resolve().parents[4]
MANIFEST = ROOT / 'LightGenV2/tasks/t06_video_quality_assessment/spatial_historical_config_import_20261006.json'


class HistoricalSpatialConfigTests(unittest.TestCase):
    def test_full_inheritance_is_present_and_matches_archived_source(self):
        rows = json.loads(MANIFEST.read_text(encoding='utf8'))['configuration_chain']
        self.assertEqual(len(rows), 28)
        self.assertEqual(sum(row['newly_tracked'] for row in rows), 14)
        for index, row in enumerate(rows):
            path = ROOT / row['path']
            raw = path.read_bytes().replace(b'\r\n', b'\n')
            self.assertEqual(hashlib.sha256(raw).hexdigest(), row['sha256_lf'], row['path'])
            if row['base_config']:
                self.assertEqual((path.parent / row['base_config']).resolve(), (ROOT / rows[index + 1]['path']).resolve())
            else:
                self.assertEqual(index, len(rows) - 1)

    def test_main_loader_retains_historical_spatial_contract(self):
        rows = json.loads(MANIFEST.read_text(encoding='utf8'))['configuration_chain']
        settings = load_settings(ROOT / rows[0]['path'], synthetic=True)
        self.assertEqual(settings.target, 'spatial')
        self.assertEqual(settings.spatial_readout_mode, 'spatial_weighted_level_residual')
        self.assertEqual(settings.spatial_residual_max, 2.0262)
        self.assertTrue(settings.strict_two_branch)


if __name__ == '__main__':
    unittest.main()
