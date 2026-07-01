import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from foundation_distillation.scripts.distillation_losses import feature_distillation_loss


def test_zero_leak_weight_records_ratio_without_changing_total_loss():
    logits = torch.randn(4, 3, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 0])
    semantic = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    teacher = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    outside = torch.tensor([0.1, 0.2, 0.3, 0.4])
    no_leak = feature_distillation_loss(logits, labels, semantic, teacher, outside, leak_loss_weight=0.0)
    weighted = feature_distillation_loss(logits, labels, semantic, teacher, outside, leak_loss_weight=2.0)
    assert torch.allclose(no_leak["leak_loss"], outside.mean())
    assert torch.allclose(no_leak["outside_camera_energy_ratio"], outside.mean())
    assert torch.allclose(weighted["total_loss"] - no_leak["total_loss"], 2.0 * outside.mean())
    no_leak["total_loss"].backward()
    assert logits.grad is not None
