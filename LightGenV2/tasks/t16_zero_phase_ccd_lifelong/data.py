"""Fixed, label-free EuroSAT RGB/SAR-to-amplitude mapping with no blank quadrant."""

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from LightGenV2.tasks.t13_four_modal_lifelong.model import normalize_power


def rgb_luma_field(images):
    """R/G/B/luma tiles preserve all input channels and fill the full aperture."""
    rgb = torch.as_tensor(np.array(images, copy=True)).float().permute(0, 3, 1, 2) / 255.0
    if rgb.shape[1] != 3:
        raise ValueError("expected three stored RGB/SAR channels")
    rgb = F.interpolate(rgb, (112, 112), mode="bilinear", align_corners=False)
    red, green, blue = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    luma = .299 * red + .587 * green + .114 * blue
    field = torch.cat((torch.cat((red, green), -1),
                       torch.cat((blue, luma), -1)), -2)
    return normalize_power(field)


class FilledEuroSatFields:
    def __init__(self, trainval: Path, holdout: Path, split: str):
        path = holdout if split == "test" else trainval
        prefix = "test" if split == "test" else ("validation" if split == "val" else "train")
        self.images = np.load(path, mmap_mode="r", allow_pickle=False)[prefix + "_images"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        scalar = np.isscalar(index)
        values = self.images[[int(index)]] if scalar else self.images[index]
        result = rgb_luma_field(values)
        return result[0] if scalar else result
