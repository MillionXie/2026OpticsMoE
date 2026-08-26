from pathlib import Path

import torch
from PIL import Image

from ..data import build_input_transform
from ..settings import load_settings


CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "release"
    / "mnist4_single_layer_17um_10cm_v2_robust_raw.yaml"
)


def test_notebook_input_encoding_is_resize_336_then_zero_pad_to_400() -> None:
    settings = load_settings(CONFIG)
    guard = (settings.input_size - settings.input_content_size) // 2
    transform = build_input_transform(settings)
    image = Image.new("L", (28, 28), color=255)
    field = transform(image)
    assert field.shape == (1, 400, 400)
    assert guard == 32
    assert torch.count_nonzero(field[:, :guard]) == 0
    assert torch.count_nonzero(field[:, -guard:]) == 0
    assert torch.count_nonzero(field[:, :, :guard]) == 0
    assert torch.count_nonzero(field[:, :, -guard:]) == 0
    assert torch.all(field[:, guard:-guard, guard:-guard] == 1.0)
