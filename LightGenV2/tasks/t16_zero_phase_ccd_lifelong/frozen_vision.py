"""One frozen CLEVR-pretrained visual front, shared by both optical architectures.

The temporary 24-way pretraining head is discarded. A deterministic 128-value
feature strip occupies the bottom quarter of each RGB tile; the upper three
quarters still contain the original RGB image. Radar/query tiles are untouched.
"""

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from LightGenV2.tasks.t09_multimodal_matching.vision import VisionEncoder
from LightGenV2.tasks.t13_four_modal_lifelong.model import normalize_power


class FrozenVisionFeatures:
    def __init__(self, checkpoint: Path, images, device, batch_size=256):
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = state.get("config", {})
        if (config.get("feature_parameters") != 32128 or
                config.get("temporary_head_parameters") != 3096 or
                config.get("test_policy") != "never read"):
            raise ValueError("expected audited full-CLEVR visual pretraining checkpoint")
        model = VisionEncoder().to(device)
        model.load_state_dict(state["model"])
        model.eval().requires_grad_(False)
        rows = []
        with torch.no_grad():
            for start in range(0, len(images), batch_size):
                raw = torch.as_tensor(np.array(images[start:start + batch_size], copy=True),
                                      device=device)
                features = model(raw)[0]
                rows.append(features.cpu().to(torch.float16))
        self.features = torch.cat(rows)
        if self.features.shape != (len(images), 128) or not torch.isfinite(self.features).all():
            raise ValueError("invalid frozen visual features")

    def select(self, indices, device):
        rows = self.features[np.asarray(indices, dtype=np.int64)].to(device=device,
                                                                       dtype=torch.float32)
        return rows


def visual_tiles(raw_images, feature_rows, device):
    """Fixed RGB/feature optical encoding; total power of three tiles is 0.5."""
    raw = torch.as_tensor(np.array(raw_images, copy=True), device=device).float()
    if raw.ndim != 4 or raw.shape[-1] != 3 or feature_rows.shape != (len(raw), 128):
        raise ValueError("expected paired RGB images and 128-value frozen features")
    rgb = F.interpolate(raw.permute(0, 3, 1, 2) / 255.0, (84, 112),
                        mode="bilinear", align_corners=False)
    glyph = F.interpolate(feature_rows.reshape(-1, 1, 8, 16), (28, 112),
                          mode="bilinear", align_corners=False)[:, 0]
    # Separate normalized power regions prevent the feature strip's magnitude
    # from suppressing raw RGB. They have no trainable parameters.
    rgb = normalize_power(rgb.flatten(2), 0.25).reshape_as(rgb)
    glyph = normalize_power(glyph, 0.25 / 3.0)
    tiles = torch.cat((rgb, glyph[:, None].expand(-1, 3, -1, -1)), dim=2)
    if tiles.shape[1:] != (3, 112, 112):
        raise AssertionError("visual tile geometry changed")
    return tiles


class FrozenVisionEuroSat:
    """Wrap audited paired locations without changing labels or SAR encoding."""

    def __init__(self, base, checkpoint, device):
        self.base = base
        self.labels = base.labels
        self.front = FrozenVisionFeatures(checkpoint, base.rgb, device)

    def __len__(self):
        return len(self.base)

    def get_batch(self, indices, device):
        indices = np.asarray(indices, dtype=np.int64)
        from .data import paired_rgb_sar_field
        original = paired_rgb_sar_field(self.base.rgb[indices], self.base.sar[indices]).to(device)
        rgb = visual_tiles(self.base.rgb[indices], self.front.select(indices, device), device)
        return torch.cat((torch.cat((rgb[:, 0], rgb[:, 1]), -1),
                          torch.cat((rgb[:, 2], original[:, 112:, 112:]), -1)), -2)

    def __getitem__(self, indices):
        # EuroSAT trainer retrieves CPU tensors and moves them to its device.
        return self.get_batch(indices, torch.device("cpu"))


class FrozenVisionClevr:
    """Keep each original image/question/label pair; replace only RGB tiles."""

    def __init__(self, base, checkpoint, device):
        self.base = base
        self.labels = base.labels
        self.front = FrozenVisionFeatures(checkpoint, base.images, device)

    def __len__(self):
        return len(self.base)

    def get_batch(self, indices, device):
        indices = np.asarray(indices, dtype=np.int64)
        original = self.base.get_batch(indices, device)
        image_indices = self.base.image_index[indices]
        rgb = visual_tiles(self.base.images[image_indices],
                           self.front.select(image_indices, device), device)
        return torch.cat((torch.cat((rgb[:, 0], rgb[:, 1]), -1),
                          torch.cat((rgb[:, 2], original[:, 112:, 112:]), -1)), -2)
