from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from LightGenV2.tasks.t06_video_quality_assessment.multivideo_settings import (
    MultiVideoGeometry,
    load_settings,
)


class MultiVideoSettingsTest(unittest.TestCase):
    def test_formal_layout_exactly_fills_active_field(self) -> None:
        geometry = MultiVideoGeometry()
        geometry.validate()
        self.assertEqual(geometry.video_origins, ((3, 3), (3, 162), (3, 321), (162, 3), (162, 162), (162, 321), (321, 3), (321, 162), (321, 321)))
        self.assertEqual(geometry.frame_origins_local, ((0, 0), (0, 79), (79, 0), (79, 79)))
        self.assertEqual(geometry.frame_expert_origins_local, ((0, 0), (0, 39), (39, 0), (39, 39)))
        self.assertEqual(geometry.video_expert_origins_local, ((0, 0), (0, 82), (82, 0), (82, 82)))

    def test_layered_candidate_preserves_nine_by_four_semantics(self) -> None:
        path = Path(__file__).parents[1] / "configs" / "lightgen" / "temporal_multivideo9x4_balanced.yaml"
        settings = load_settings(path)
        self.assertEqual(settings.videos_per_field, 9)
        self.assertEqual(settings.frame_count, 4)
        self.assertEqual(settings.geometry.active_size, 478)
        self.assertGreaterEqual(settings.unmodulated_power_fraction_min, 0.20)
        self.assertEqual(settings.phase_snapshot_interval_epochs, 5)


if __name__ == "__main__":
    unittest.main()
