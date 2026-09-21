import json
from pathlib import Path

import numpy as np
import torch

from LightGenV2.tasks.t13_four_modal_lifelong.data import PhysicalTextFields, _rgb_field


def test_rgb_uses_all_channels_and_has_unit_power():
    x = np.zeros((2, 56, 56, 3), np.uint8)
    x[0, :, :, 0] = 255
    x[1, :, :, 2] = 255
    field = _rgb_field(x)
    assert field.shape == (2, 224, 224)
    assert not torch.equal(field[0], field[1])
    assert torch.allclose(field.square().sum((-2, -1)), torch.ones(2), atol=1e-5)


def test_physical_video_text_is_balanced_and_uses_both_modalities(tmp_path: Path):
    fields = np.zeros((2, 224, 224), np.float16)
    fields[0, :112] = 1
    fields[1, 112:] = 1
    np.savez(tmp_path / "train.npz", fields=fields, labels=np.array([0, 1]))
    dataset = PhysicalTextFields(tmp_path / "train.npz")
    assert dataset.labels().tolist() == [1, 0, 0, 1]
    batch = dataset[np.arange(4)]
    assert batch.shape == (4, 224, 224)
    assert torch.allclose(batch.square().sum((-2, -1)), torch.ones(4), atol=1e-4)
    assert not torch.equal(batch[0, :, 112:], batch[1, :, 112:])

