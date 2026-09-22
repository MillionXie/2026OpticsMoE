from __future__ import annotations

import ast
import unittest
from pathlib import Path

import torch
import yaml

from LightGenV2.tasks.t06_video_quality_assessment.deepseek_vl2_quality import (
    FiveQualityRows,
    TARGETS,
    build_conversation,
    quality_token_rows,
    validate_config,
)


TASK_DIR = Path(__file__).parents[1]
CONFIG = TASK_DIR / "configs" / "baselines" / "deepseek_vl2_tiny_lgvq_4f_r384.yaml"


class _Tokenizer:
    def __init__(self, mapping: dict[str, list[int]]) -> None:
        self.mapping = mapping

    def encode(self, word: str, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return self.mapping[word]


class DeepSeekVLV2QualityContractTest(unittest.TestCase):
    def test_config_freezes_backbone_and_matches_qwen_protocol(self) -> None:
        raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        validate_config(raw)
        self.assertTrue(raw["model"]["backbone_frozen"])
        self.assertEqual(raw["model"]["trainable_parameters"], 6400)
        self.assertEqual(raw["input"]["frame_count"], 4)
        self.assertEqual(raw["input"]["frame_fractions"], [0.10, 0.37, 0.63, 0.90])
        self.assertEqual(tuple(raw["task"]["targets"]), TARGETS)
        self.assertEqual(raw["training"]["epochs"], 50)
        self.assertEqual(raw["training"]["batch_size"], 512)

    def test_conversation_has_ordered_four_frame_placeholders(self) -> None:
        conversation = build_conversation("temporal", 4)
        user = conversation[0]
        self.assertEqual(user["role"], "<|User|>")
        self.assertEqual(user["content"].count("<image>"), 4)
        self.assertLess(user["content"].index("Frame 1"), user["content"].index("Frame 4"))
        self.assertIn("temporal quality", user["content"])
        self.assertEqual(len(user["images"]), 4)

    def test_only_five_quality_rows_are_trainable(self) -> None:
        head = FiveQualityRows(torch.zeros(5, 1280))
        self.assertEqual(sum(parameter.numel() for parameter in head.parameters()), 6400)
        self.assertEqual(tuple(head(torch.ones(3, 1280)).shape), (3, 5))

    def test_native_rows_support_explicit_split_token_fallback(self) -> None:
        mapping = {
            "Bad": [1],
            "Poor": [2, 3],
            "Fair": [4],
            "Good": [5],
            "Excellent": [6],
        }
        weight = torch.arange(10 * 4, dtype=torch.float32).reshape(10, 4)
        rows, token_ids, mode = quality_token_rows(_Tokenizer(mapping), weight)
        self.assertEqual(token_ids[1], [2, 3])
        torch.testing.assert_close(rows[1], weight[[2, 3]].mean(0))
        self.assertEqual(mode, "mean_native_subtoken_rows")

    def test_source_has_explicit_full_backbone_freeze_and_two_checkpoints(self) -> None:
        path = TASK_DIR / "deepseek_vl2_quality.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertIn(".eval().requires_grad_(False)", source)
        self.assertIn('"best_checkpoint.pt"', source)
        self.assertIn('"last_checkpoint.pt"', source)
        self.assertNotIn("periodic_checkpoint", source)
        self.assertGreater(len(list(ast.walk(tree))), 500)


if __name__ == "__main__":
    unittest.main()
