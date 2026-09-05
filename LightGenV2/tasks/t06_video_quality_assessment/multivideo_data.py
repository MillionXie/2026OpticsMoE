"""Nine-video physical-field grouping for T06 Temporal VQA."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch.utils.data import Dataset


class NineVideoFieldDataset(Dataset[dict[str, Any]]):
    """Groups unrelated videos into one physical 3x3 optical field.

    Group membership is regenerated from ``grouping_seed`` every epoch.  A
    validity mask makes the general contract safe when a future dataset is not
    divisible by nine; the current 2250/558 LGVQ splits require no padding.
    """

    def __init__(
        self,
        payload: Mapping[str, Any],
        split: str,
        *,
        videos_per_field: int = 9,
        grouping_seed: int = 0,
        shuffle_membership: bool,
    ) -> None:
        self.payload = payload
        self.videos_per_field = int(videos_per_field)
        indices = torch.tensor(
            [index for index, value in enumerate(payload["splits"]) if value == split],
            dtype=torch.long,
        )
        if not len(indices):
            raise RuntimeError(f"No samples for split={split}")
        if shuffle_membership:
            generator = torch.Generator().manual_seed(int(grouping_seed))
            indices = indices[torch.randperm(len(indices), generator=generator)]
        remainder = len(indices) % self.videos_per_field
        valid = torch.ones(len(indices), dtype=torch.bool)
        if remainder:
            padding = self.videos_per_field - remainder
            indices = torch.cat((indices, indices[:padding]))
            valid = torch.cat((valid, torch.zeros(padding, dtype=torch.bool)))
        self.groups = indices.reshape(-1, self.videos_per_field)
        self.valid = valid.reshape(-1, self.videos_per_field)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source = self.groups[index]
        item: dict[str, Any] = {
            "vision_tokens": self.payload["vision_tokens"][source].float(),
            "quality_tokens": self.payload["quality_tokens"][source].float(),
            "language_tokens": self.payload["language_tokens"][0].float(),
            "language_mask": self.payload["language_mask"][0].bool(),
            "target": self.payload["targets"][source].float(),
            "source_indices": source,
            "valid": self.valid[index],
        }
        if "soft_target_present" in self.payload:
            item["soft_target"] = self.payload["soft_targets"][source].float()
            item["soft_target_present"] = self.payload["soft_target_present"][source]
        return item


def permute_video_slots(
    batch: Mapping[str, Any], *, generator: torch.Generator | None = None
) -> tuple[dict[str, Any], torch.Tensor]:
    """Independently permute the nine optical slots of each physical field."""

    size = int(batch["target"].shape[1])
    permutations = torch.stack(
        [torch.randperm(size, generator=generator) for _ in range(batch["target"].shape[0])]
    )
    inverse = torch.argsort(permutations, dim=1)
    result = dict(batch)
    for name in (
        "vision_tokens",
        "quality_tokens",
        "target",
        "source_indices",
        "valid",
        "soft_target",
        "soft_target_present",
    ):
        value = batch.get(name)
        if not torch.is_tensor(value) or value.ndim < 2 or value.shape[1] != size:
            continue
        shape = (value.shape[0], size) + (1,) * (value.ndim - 2)
        result[name] = torch.gather(
            value, 1, permutations.reshape(shape).expand_as(value)
        )
    return result, inverse


__all__ = ["NineVideoFieldDataset", "permute_video_slots"]
