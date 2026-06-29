import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from foundation_distillation.scripts.distillation_losses import feature_distillation_loss


def test_feature_distillation_loss_backpropagates():
    logits = torch.randn(4, 10, requires_grad=True)
    projected = torch.randn(4, 32, requires_grad=True)
    teacher = torch.randn(4, 32)
    losses = feature_distillation_loss(logits, torch.tensor([0, 1, 2, 3]), projected, teacher)
    losses["total_loss"].backward()
    assert logits.grad is not None
    assert projected.grad is not None
    assert 0.0 <= float(losses["feature_loss"].detach()) <= 2.0
