from __future__ import annotations

import ast
import unittest
from pathlib import Path

import yaml


TASK_DIR = Path(__file__).parents[1]
CONFIG = TASK_DIR / "configs" / "baselines" / "qwen3vl_quality_tokens_r448.yaml"
SPATIAL_CONFIG = (
    TASK_DIR
    / "configs"
    / "baselines"
    / "qwen3vl_spatial_quality_tokens_4f_r448.yaml"
)
EXPECTED_PROMPT = (
    "Please evaluate the temporal quality of this video and rate it using one of "
    "the following five levels: Excellent, Good, Fair, Poor, or Bad."
)
EXPECTED_FRACTIONS = {
    4: [0.10, 0.37, 0.63, 0.90],
    9: [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
    16: [0.10 + index * 0.80 / 15.0 for index in range(16)],
}


class QualityTokenResolutionContractTest(unittest.TestCase):
    def test_formal_resolution_and_frame_contract(self) -> None:
        raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(raw["input"]["image_size"], 448)
        self.assertEqual(raw["input"]["qwen3vl_patch_size"], 16)
        self.assertEqual(raw["input"]["qwen3vl_spatial_merge_size"], 2)
        self.assertEqual(raw["input"]["qwen3vl_temporal_patch_size"], 2)
        self.assertEqual(raw["input"]["effective_spatial_stride"], 32)
        self.assertEqual(raw["input"]["frame_counts"], [4, 9, 16])
        self.assertEqual(raw["task"]["prompt"], EXPECTED_PROMPT)
        self.assertEqual(raw["timing"]["explicit_warmup_forwards"], 0)
        self.assertEqual(raw["timing"]["pretest_inference_forwards"], 0)
        self.assertTrue(raw["timing"]["include_first_test_video"])
        self.assertEqual((448 // 16) ** 2, 784)
        self.assertEqual((448 // 32) ** 2, 196)
        self.assertEqual(2 * 784, 1568)  # four input frames before merger
        self.assertEqual(5 * 784, 3920)  # nine frames are padded to five temporal pairs
        self.assertEqual(8 * 784, 6272)  # sixteen frames before merger
        for count in (4, 9, 16):
            configured = raw["input"]["frame_fractions"][count]
            self.assertEqual(len(configured), count)
            for left, right in zip(configured, EXPECTED_FRACTIONS[count]):
                self.assertAlmostEqual(left, right, places=8)

    def test_quality_head_has_only_five_output_rows(self) -> None:
        source = (TASK_DIR / "quality_token_common.py").read_text(encoding="utf-8")
        self.assertIn("rows.shape != (5, 2048)", source)
        self.assertIn("value.float() @ self.weight.t()", source)
        self.assertEqual(5 * 2048, 10240)

    def test_spatial_baseline_is_four_frame_and_not_temporal_prompt(self) -> None:
        raw = yaml.safe_load(SPATIAL_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(raw["task"]["target"], "spatial_mos")
        self.assertEqual(raw["input"]["frame_counts"], [4])
        self.assertIn("spatial quality", raw["task"]["prompt"])
        self.assertNotIn("temporal quality", raw["task"]["prompt"])
        self.assertEqual(raw["model"]["trainable_parameters"], 10240)

    def test_dataset_once_source_has_no_warmup_loop(self) -> None:
        path = TASK_DIR / "quality_token_dataset_once.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parser_text = path.read_text(encoding="utf-8")
        self.assertNotIn("--warmup", parser_text)
        self.assertIn('"explicit_warmup_forwards": 0', parser_text)
        self.assertIn('"pretest_inference_forwards": 0', parser_text)
        self.assertGreater(len(list(ast.walk(tree))), 100)

    def test_formal_checkpoint_names_are_not_periodic(self) -> None:
        source = (TASK_DIR / "quality_token_train.py").read_text(encoding="utf-8")
        self.assertIn('"best_checkpoint.pt"', source)
        self.assertIn('"last_checkpoint.pt"', source)
        self.assertNotIn("phase_snapshot", source)


if __name__ == "__main__":
    unittest.main()
