from __future__ import annotations

import torch

from LightGenV2.tasks.t12_text_to_image.small_fullframe import (
    SmallEditorConfig,
    SmallFullFrameEditor,
    encode_prompts,
)


def test_small_editor_is_under_50m_and_generates_full_frame() -> None:
    config = SmallEditorConfig(image_size=32, widths=(16, 24, 32, 48), condition_dim=32, text_width=16)
    model = SmallFullFrameEditor(config)
    assert sum(parameter.numel() for parameter in model.parameters()) < 50_000_000
    reference = torch.randn(2, 3, 32, 32).clamp(-1, 1)
    tokens = encode_prompts(["warm studio", "cool studio"], config.max_text_bytes, torch.device("cpu"))
    output = model(reference, tokens, torch.randn_like(reference))
    assert output.shape == reference.shape
    assert not torch.equal(output, reference)
    assert float(model.bottleneck.fusion.alpha) >= .4
