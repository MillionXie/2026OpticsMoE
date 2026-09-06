from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from LightGenV2.tasks.t06_video_quality_assessment.project import (
    CURRENT_PROFILE,
    inspect_profile,
    materialize_launch_config,
)


class Temporal36ProjectContractTest(unittest.TestCase):
    def test_profile_points_to_existing_verified_backend(self) -> None:
        report = inspect_profile(CURRENT_PROFILE)
        self.assertTrue(report["backend_config_present"])
        self.assertEqual(
            report["canonical_checkpoint_expected_sha256"],
            "159b1d8cd31aa5f817d274f2930129601d4f0a365f01c430a8fefcc5989c8730",
        )

    def test_materialized_config_owns_new_run_and_preserves_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            _, config, package = materialize_launch_config(CURRENT_PROFILE, run)
            raw = yaml.safe_load(config.read_text(encoding="utf-8"))
            self.assertEqual(Path(raw["output_dir"]), run.resolve())
            self.assertEqual(raw["model"]["frame_count"], 36)
            self.assertEqual(raw["geometry"]["lane_grid"], 6)
            self.assertTrue(Path(raw["data"]["vision_cache"]).is_absolute())
            self.assertEqual(
                package,
                "experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54",
            )

    def test_spatial_single_video4_profile_is_not_multivideo_reuse(self) -> None:
        profile = "spatial_single_video4_balanced"
        report = inspect_profile(profile)
        self.assertTrue(report["backend_config_present"])
        self.assertEqual(
            report["canonical_checkpoint_expected_sha256"],
            "aa1e28d42995d2187b9c949a49d7c6d37891435f1eb3b7d2fe1ecef5b35d50e8",
        )
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "spatial_run"
            raw_profile, config, package = materialize_launch_config(profile, run)
            raw = yaml.safe_load(config.read_text(encoding="utf-8"))
            self.assertEqual(raw["task"]["target_name"], "spatial")
            self.assertEqual(raw["model"]["frame_count"], 4)
            self.assertEqual(raw["geometry"]["lane_grid"], 2)
            self.assertEqual(raw["router"]["top_k"], 2)
            self.assertFalse(raw_profile["frame_semantics"]["multi_video_reuse"])
            self.assertEqual(
                package,
                "experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54",
            )


if __name__ == "__main__":
    unittest.main()
