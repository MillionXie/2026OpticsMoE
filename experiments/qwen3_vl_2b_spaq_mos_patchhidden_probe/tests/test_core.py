from __future__ import annotations

import torch
from torch import nn

from experiments.qwen3_vl_2b_spaq_mos_patchhidden_probe.capture import PatchHiddenCapture
from experiments.qwen3_vl_2b_spaq_mos_patchhidden_probe.settings import load_settings
from experiments.qwen3_vl_2b_spaq_mos_patchhidden_probe.training import build_head


CONFIG = "experiments/qwen3_vl_2b_spaq_mos_patchhidden_probe/configs/spaq_mos_patchhidden_probe_smoke.json"


def test_config_and_head() -> None:
    settings = load_settings(CONFIG)
    head = build_head(1024, settings)
    assert head(torch.randn(4, 1024)).shape == (4,)
    assert sum(parameter.numel() for parameter in head.parameters()) == 3073


def test_capture_is_identity_and_preserves_boundaries() -> None:
    module = PatchHiddenCapture()
    hidden = torch.randn(7, 8, requires_grad=True)
    output = module(hidden, cu_seqlens=torch.tensor([0, 3, 7], dtype=torch.int32))
    assert output is hidden
    assert module.last_token_counts == [3, 4]
    assert torch.equal(module.last_hidden, hidden.detach())
    output.square().mean().backward()
    assert hidden.grad is not None


def test_head_backward() -> None:
    settings = load_settings(CONFIG); head = build_head(8, settings)
    prediction = head(torch.randn(6, 8))
    nn.SmoothL1Loss(beta=0.1)(prediction, torch.rand(6)).backward()
    assert all(parameter.grad is not None for parameter in head.parameters())

