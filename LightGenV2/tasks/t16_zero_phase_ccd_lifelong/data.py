"""Paired, label-free RGB/SAR amplitude encoding for the EuroSAT task."""

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from LightGenV2.tasks.t13_four_modal_lifelong.model import normalize_power


def paired_rgb_sar_field(rgb_images, sar_images):
    """Tile R/G/B/SAR; give RGB and SAR equal incident optical power.

    The SAR tile is the arithmetic mean of the source's independently
    normalized VV and VH channels. No learned electronic encoder is used.
    """
    rgb = torch.as_tensor(np.array(rgb_images, copy=True)).float()
    sar = torch.as_tensor(np.array(sar_images, copy=True)).float()
    if rgb.ndim != 4 or rgb.shape != sar.shape or rgb.shape[-1] != 3:
        raise ValueError("expected paired BxHxWx3 RGB and SAR arrays")
    rgb = F.interpolate(rgb.permute(0, 3, 1, 2) / 255.0, (112, 112),
                        mode="bilinear", align_corners=False)
    sar = F.interpolate(sar.permute(0, 3, 1, 2) / 255.0, (112, 112),
                        mode="bilinear", align_corners=False)
    radar = (sar[:, 0] + sar[:, 1]) / 2
    rgb_tiles = normalize_power(rgb.flatten(2), 0.5).reshape_as(rgb)
    radar = normalize_power(radar, 0.5)
    return torch.cat((torch.cat((rgb_tiles[:, 0], rgb_tiles[:, 1]), -1),
                      torch.cat((rgb_tiles[:, 2], radar), -1)), -2)


class PairedEuroSatFields:
    """Read the audited RGB/SAR pairs without changing their spatial split."""

    def __init__(self, trainval: Path, holdout: Path, split: str):
        if split not in {"train", "val", "test"}:
            raise ValueError(split)
        path = holdout if split == "test" else trainval
        prefix = "test" if split == "test" else ("validation" if split == "val" else "train")
        with np.load(path, allow_pickle=False) as data:
            images = data[prefix + "_images"]
            labels = data[prefix + "_labels"]
            domains = data[prefix + "_domains"]
            ids = data[prefix + "_ids"]
        if len(images) % 2 or not (len(images) == len(labels) == len(domains) == len(ids)):
            raise ValueError("EuroSAT records must be complete RGB/SAR pairs")
        if not np.array_equal(domains.reshape(-1, 2),
                              np.tile([0, 1], (len(images) // 2, 1))):
            raise ValueError("expected RGB then SAR for each location")
        if not np.array_equal(labels[::2], labels[1::2]):
            raise ValueError("paired modalities have different labels")
        rgb_ids = np.char.partition(ids[::2].astype(str), ":")[:, 0]
        sar_ids = np.char.partition(ids[1::2].astype(str), ":")[:, 0]
        if not np.array_equal(rgb_ids, sar_ids):
            raise ValueError("RGB/SAR location IDs do not match")
        self.rgb = images[::2]
        self.sar = images[1::2]
        self.labels = labels[::2].astype(np.int64, copy=False)
        self.pair_ids = rgb_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        scalar = np.isscalar(index)
        selection = [int(index)] if scalar else index
        field = paired_rgb_sar_field(self.rgb[selection], self.sar[selection])
        return field[0] if scalar else field
