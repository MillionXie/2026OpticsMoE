import numpy as np
import torch

from LightGenV2.tasks.t16_zero_phase_ccd_lifelong.frozen_vision import visual_tiles


def test_visual_tiles_keep_raw_rgb_and_feature_strip_with_fixed_power():
    raw = np.zeros((2, 16, 16, 3), dtype=np.uint8)
    raw[:, :, :, 0] = 100
    raw[:, :, :, 1] = 150
    raw[:, :, :, 2] = 200
    features = torch.ones(2, 128)
    tiles = visual_tiles(raw, features, torch.device("cpu"))
    assert tiles.shape == (2, 3, 112, 112)
    assert torch.all(tiles[:, :, :84] > 0)
    assert torch.all(tiles[:, :, 84:] > 0)
    assert torch.allclose(tiles[:, :, :84].square().sum((1, 2, 3)),
                          torch.full((2,), 0.25), atol=1e-5)
    assert torch.allclose(tiles[:, :, 84:].square().sum((1, 2, 3)),
                          torch.full((2,), 0.25), atol=1e-5)
