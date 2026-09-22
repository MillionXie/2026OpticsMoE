from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image, ImageDraw
from torch import nn

from LightGenV2.tasks.t12_text_to_image.compact_product_model import (
    COMPACT_ATTENTION_PRUNE_SPEC,
    IdentityResidualOpticalMidBlock,
)
from LightGenV2.tasks.t12_text_to_image.product_repair_model import RepairModelConfig

from LightGenV2.tasks.t12_text_to_image.half_qwen import retain_language_layers
from LightGenV2.tasks.t12_text_to_image.product_scene_replace_data import (
    COMBINATIONS,
    HOLDOUT_COMBINATIONS,
    TRAIN_COMBINATIONS,
    ProductBackgroundReplacementDataset,
    instruction_rows,
    render_composed_background,
)


def _fixture(root: Path) -> Path:
    rows = instruction_rows()
    cache = root / "instructions.pt"
    torch.save({"rows": rows, "text": torch.randn(len(rows), 32)}, cache)
    for split in ("train", "val", "test"):
        image_dir = root / "images" / split / "lamp"
        mask_dir = root / "masks" / split / "lamp"
        image_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        image = Image.new("RGB", (64, 64), "white")
        ImageDraw.Draw(image).rectangle((24, 9, 40, 56), fill=(28, 28, 30))
        mask = Image.new("L", (64, 64), 0)
        ImageDraw.Draw(mask).rectangle((24, 9, 40, 56), fill=255)
        image.save(image_dir / "lamp.jpg")
        mask.save(mask_dir / "lamp.png")
        manifest = {
            "sample_id": f"lamp-{split}", "sequence_id": f"identity-{split}",
            "category": "lamp", "caption": "a lamp",
            "image_path": f"images/{split}/lamp/lamp.jpg",
            "mask_path": f"masks/{split}/lamp/lamp.png",
            "license": "CC BY 4.0", "source_url": "https://example.test/abo",
        }
        (root / f"{split}.jsonl").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    return cache


def test_half_qwen_physically_removes_second_half() -> None:
    language = nn.Module()
    language.layers = nn.ModuleList([nn.Linear(3, 3, bias=False) for _ in range(28)])
    language.config = SimpleNamespace(num_hidden_layers=28)
    report = retain_language_layers(language, 14)
    assert len(language.layers) == 14
    assert language.config.num_hidden_layers == 14
    assert report["language_transformer_depth_fraction"] == 0.5
    assert report["language_transformer_parameters_retained"] == report["language_transformer_parameters_removed"]


def test_replacement_input_already_has_a_different_background(tmp_path: Path) -> None:
    cache = _fixture(tmp_path)
    dataset = ProductBackgroundReplacementDataset(tmp_path, "train", 64, cache)
    value = dataset[0]
    assert not torch.equal(value["reference"], value["target"])
    assert value["source_scene_id"] != value["target_scene_id"]
    assert not value["held_out_combination"]
    assert "white background" not in value["prompt"]
    assert torch.allclose(value["foreground_mask"] + value["background_mask"], torch.ones(1, 64, 64))


def test_exact_combinations_are_held_out_but_all_attributes_are_seen(tmp_path: Path) -> None:
    cache = _fixture(tmp_path)
    validation = ProductBackgroundReplacementDataset(tmp_path, "val", 64, cache)
    assert len(COMBINATIONS) == 48
    assert len(TRAIN_COMBINATIONS) == 40
    assert len(HOLDOUT_COMBINATIONS) == 8
    assert all(validation[index]["held_out_combination"] for index in range(len(validation)))
    for axis in range(4):
        assert {value[axis] for value in TRAIN_COMBINATIONS} == {value[axis] for value in COMBINATIONS}


def test_composed_background_is_deterministic_and_attribute_sensitive() -> None:
    cool_dim_left = ("modern_study", "cool", "dim", "left")
    warm_bright_right = ("modern_study", "warm", "bright", "right")
    first = render_composed_background(cool_dim_left, 64, "lamp", "target")
    repeat = render_composed_background(cool_dim_left, 64, "lamp", "target")
    changed = render_composed_background(warm_bright_right, 64, "lamp", "target")
    assert first.tobytes() == repeat.tobytes()
    assert first.tobytes() != changed.tobytes()


def test_compact_pruning_removes_all_but_two_deep_attention_modules() -> None:
    assert len(COMPACT_ATTENTION_PRUNE_SPEC) == 7
    assert {tuple((item["side"], item["block"])) for item in COMPACT_ATTENTION_PRUNE_SPEC} == {
        ("down", 0), ("down", 1), ("up", 0), ("up", 1), ("up", 2),
    }


def test_identity_optical_mid_has_no_electronic_transform() -> None:
    block = IdentityResidualOpticalMidBlock(
        channels=8, timestep_dim=12, condition_dim=10,
        config=RepairModelConfig(
            optical_width=8, optical_grid=2, optical_experts=2, optical_top_k=1,
            alpha_initial=0.5, alpha_minimum=0.4, alpha_maximum=0.75,
        ),
    )
    output = block(torch.randn(2, 8, 4, 4), torch.randn(2, 12), torch.randn(2, 3, 10))
    assert output.shape == (2, 8, 4, 4)
    report = block.architecture_report()
    assert "identity residual" in report["electronic_branch"]
    assert report["physical_latency_ms"] == 1.0447 * 6
