from pathlib import Path

import torch

from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.modeling import _build_pose_head, DeconvPoseHead
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.settings import load_settings


def test_qwen_head_budget_and_backward():
    cfg = Path(__file__).resolve().parents[1]/'configs/qwen_deconv40.yaml'
    settings = load_settings(cfg)
    small = _build_pose_head(1024, settings)
    large = DeconvPoseHead(1024)
    assert sum(p.numel() for p in small.parameters()) == 138422
    assert sum(p.numel() for p in large.parameters()) == 1102990
    assert len(small.body) == len(large.body) == 2
    assert len(small.upsampler) == len(large.upsampler) == 2
    x = torch.randn(2, 1024, 14, 14)
    y = small(x)
    assert y.shape == (2, 14, 56, 56)
    y.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in small.parameters())
    assert settings.teacher_epochs == 40
    assert settings.teacher_batch_size == 8
    assert settings.teacher_learning_rate == .001
    assert settings.coordinate_loss_weight == .1
