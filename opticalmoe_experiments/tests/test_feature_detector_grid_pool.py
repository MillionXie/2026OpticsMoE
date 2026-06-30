import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.optics.distilled_moe import GridPoolFeatureDetector


def test_grid_pool_feature_detector_uses_detector_intensity():
    detector = GridPoolFeatureDetector(grid_size=16, pooling="sum", normalize_total_energy=True)
    feature = detector(torch.rand(2, 520, 520))
    assert feature.shape == (2, 256)
    assert torch.allclose(feature.sum(dim=1), torch.ones(2), atol=1e-5)
