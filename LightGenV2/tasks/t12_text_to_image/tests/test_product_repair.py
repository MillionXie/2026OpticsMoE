from __future__ import annotations

import gzip
import io
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch import nn

from LightGenV2.tasks.t12_text_to_image.product_repair_data import (
    ProductRepairDataset,
    instruction_rows,
)
from LightGenV2.tasks.t12_text_to_image.product_repair_model import (
    ParallelOpticalDecoderBlock,
    RepairModelConfig,
    expand_reference_conditioning,
)


def _dataset_fixture(root: Path) -> tuple[Path, Path]:
    data = root / "data"
    image_dir = data / "images" / "train"
    image_dir.mkdir(parents=True)
    image = Image.new("RGB", (64, 64), "white")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((12, 8, 52, 56), radius=8, fill=(60, 90, 170))
    image.save(image_dir / "pillow.jpg")
    row = {
        "sample_id": "pillow-test-00", "sequence_id": "pillow-test",
        "category": "pillow", "caption": "a blue pillow",
        "image_path": "images/train/pillow.jpg", "license": "CC BY 4.0",
        "source_url": "https://example.test", "source_archive": "fixture",
        "source_member": "fixture.png", "source_title": "fixture", "modified": "fixture",
    }
    for split in ("train", "val", "test"):
        (data / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    rows = instruction_rows()
    cache = root / "instructions.pt"
    torch.save({"rows": rows, "text": torch.randn(len(rows), 32)}, cache)
    return data, cache


def test_product_repair_uses_text_selected_target(tmp_path: Path) -> None:
    data, cache = _dataset_fixture(tmp_path)
    dataset = ProductRepairDataset(data, "train", 64, cache, seed=3)
    sample = dataset[0]
    counterfactual = dataset[1]
    assert len(dataset) == 2
    assert sample["reference"].shape == (3, 64, 64)
    assert sample["target"].shape == (3, 64, 64)
    assert sample["selected_region"] != sample["distractor_region"]
    selected_change = ((sample["target"] - sample["reference"]).abs() * sample["selected_mask"]).sum()
    distractor_change = ((sample["target"] - sample["reference"]).abs() * sample["distractor_mask"]).sum()
    assert float(selected_change) > 0
    assert float(distractor_change) < float(selected_change) * 0.15
    assert "only" in sample["prompt"] or "just" in sample["prompt"]
    torch.testing.assert_close(sample["reference"], counterfactual["reference"])
    assert sample["selected_region"] == counterfactual["distractor_region"]
    assert sample["distractor_region"] == counterfactual["selected_region"]
    assert sample["prompt"] != counterfactual["prompt"]
    assert not torch.equal(sample["target"], counterfactual["target"])


class _UpBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnets = nn.ModuleList([nn.Conv2d(4, 4, 1)])
        self.upsamplers = nn.ModuleList()

    def forward(self, hidden_states, **kwargs):
        del kwargs
        return torch.nn.functional.interpolate(hidden_states, scale_factor=2, mode="nearest")


def test_decoder_optics_are_parallel_and_alpha_is_large() -> None:
    block = ParallelOpticalDecoderBlock(
        _UpBlock(), input_channels=4, output_channels=4,
        timestep_dim=6, condition_dim=5,
        config=RepairModelConfig(
            optical_width=8, optical_grid=2, optical_experts=2, optical_top_k=1
        ),
    )
    value = torch.randn(2, 4, 4, 4)
    result = block(
        value, (torch.randn_like(value),), torch.randn(2, 6),
        encoder_hidden_states=torch.randn(2, 3, 5),
    )
    assert result.shape == (2, 4, 8, 8)
    assert 0.4 <= float(block.fusion.alpha) <= 0.75
    assert block.architecture_report()["electronic_and_optical_are_parallel"] is True


def test_reference_conditioning_preserves_pretrained_channels() -> None:
    unet = nn.Module()
    unet.conv_in = nn.Conv2d(4, 8, 3, padding=1)
    unet.config = type("Config", (), {"in_channels": 4})()
    original = unet.conv_in.weight.detach().clone()
    expanded = expand_reference_conditioning(unet)
    torch.testing.assert_close(expanded.weight[:, :4], original)
    assert torch.count_nonzero(expanded.weight[:, 4:]) == 0
    assert expanded.in_channels == 8
