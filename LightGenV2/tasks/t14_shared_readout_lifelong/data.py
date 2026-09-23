"""Raw, task-dependent fields without learned electronic encoders.

Speech/video candidate order is deterministic per example but changes across
examples. The target is the *position* of the correct word/caption, forcing
sample-specific text to matter to the fixed shared ten-output readout.
"""

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from LightGenV2.tasks.t13_four_modal_lifelong.data import (
    PhysicalRank10Fields, SpeechRank8Fields, _fixed_text, normalize_power,
)


def candidate_permutations(count: int, classes: int, seed: int):
    rng = np.random.default_rng(seed)
    return np.stack([rng.permutation(classes) for _ in range(count)]).astype(np.int16)


def position_labels(permutations: np.ndarray, original_labels: np.ndarray):
    original_labels = np.asarray(original_labels, dtype=np.int64)
    if len(permutations) != len(original_labels):
        raise ValueError("one permutation is needed per example")
    matched = permutations == original_labels[:, None]
    if not np.all(matched.sum(1) == 1):
        raise ValueError("candidate order must contain the original label exactly once")
    return matched.argmax(1).astype(np.int64)


class ClevrRawPairs:
    """All original CLEVR images, only one positive and one negative query each."""

    def __init__(self, source: Path, split: str):
        source = Path(source)
        self.images = np.load(source / f"{split}_images.npy", mmap_mode="r")
        image_index = np.load(source / f"{split}_image_index.npy", mmap_mode="r")
        all_labels = np.load(source / f"{split}_labels.npy", mmap_mode="r")
        token_ids = np.load(source / f"{split}_token_ids.npy", mmap_mode="r")
        if len(all_labels) != len(self.images) * 6:
            raise ValueError("expected the six-query full-image source")
        chosen = (np.arange(len(self.images), dtype=np.int64)[:, None] * 6
                  + np.array([0, 1], dtype=np.int64)).reshape(-1)
        self.image_index = np.asarray(image_index[chosen], dtype=np.int64)
        self.labels = np.asarray(all_labels[chosen], dtype=np.int64)
        self.token_ids = np.asarray(token_ids[chosen], dtype=np.uint8)
        assert np.array_equal(self.image_index.reshape(-1, 2)[:, 0],
                              self.image_index.reshape(-1, 2)[:, 1])
        assert np.all(self.labels.reshape(-1, 2) == [1, 0])
        assert len(np.unique(self.image_index)) == len(self.images)

    def __len__(self):
        return len(self.labels)

    def get_batch(self, indices, device):
        indices = np.asarray(indices, dtype=np.int64)
        raw = np.array(self.images[self.image_index[indices]], copy=True)
        rgb = torch.as_tensor(raw, device=device).float().permute(0, 3, 1, 2) / 255.0
        rgb = F.interpolate(rgb, (112, 112), mode="bilinear", align_corners=False)
        rgb = rgb * (0.5 / rgb.square().sum((1, 2, 3), keepdim=True).clamp_min(1e-20)).sqrt()
        ids = torch.as_tensor(np.array(self.token_ids[indices], copy=True),
                              device=device, dtype=torch.long)
        text = F.interpolate(_fixed_text(ids)[:, None], (112, 112),
                             mode="nearest")[:, 0]
        text = normalize_power(text, 0.5)
        return torch.cat((torch.cat((rgb[:, 0], rgb[:, 1]), -1),
                          torch.cat((rgb[:, 2], text), -1)), -2)


class SpeechPermutedCandidates:
    def __init__(self, source: Path, split: str, seed: int = 17):
        self.base = SpeechRank8Fields(source, split)
        self.original_labels = np.asarray([row["audio_class"] for row in self.base.rows],
                                          dtype=np.int64)
        self.permutations = candidate_permutations(len(self.base), 8, seed)
        self.labels = position_labels(self.permutations, self.original_labels)
        self.original_bank = self.base.candidate_bank.reshape(8, 28, 112)

    def __len__(self):
        return len(self.base)

    def get_batch(self, indices, device):
        indices = np.asarray(indices, dtype=np.int64)
        left = self.base[indices][:, :, :112].to(device)
        order = torch.as_tensor(self.permutations[indices], device=device, dtype=torch.long)
        bank = self.original_bank.to(device)[order].reshape(-1, 224, 112)
        return torch.cat((left, bank), -1)


class PhysicalPermutedCandidates:
    def __init__(self, roots: dict, split: str, seed: int = 17):
        self.base = PhysicalRank10Fields(roots, split)
        self.original_labels = self.base.labels()
        self.permutations = candidate_permutations(len(self.base), 10, seed)
        self.labels = position_labels(self.permutations, self.original_labels)
        tokens = torch.zeros(10, 6, dtype=torch.long)
        for concept in range(5):
            tokens[2 * concept:2 * concept + 2, 0] = 2 + concept
            tokens[2 * concept, 1] = 7
            tokens[2 * concept + 1, 1] = 8
        encoded = _fixed_text(tokens)
        edges = np.linspace(0, 224, 11).round().astype(int)
        self.bands = [torch.stack([
            F.interpolate(encoded[k][None, None],
                          (int(edges[pos + 1] - edges[pos]), 112),
                          mode="nearest")[0, 0]
            for k in range(10)]) for pos in range(10)]

    def __len__(self):
        return len(self.base)

    def get_batch(self, indices, device):
        indices = np.asarray(indices, dtype=np.int64)
        left = self.base[indices][:, :, :112].to(device)
        order = torch.as_tensor(self.permutations[indices], device=device, dtype=torch.long)
        bank = torch.cat([band.to(device)[order[:, pos]]
                          for pos, band in enumerate(self.bands)], dim=1)
        bank = normalize_power(bank, 0.5)
        return normalize_power(torch.cat((left, bank), -1))
