from __future__ import annotations

import torch
from torch.utils.data import Sampler


class EpochRotatingSampler(Sampler[int]):
    """Deterministic shuffled epoch windows that eventually cover the retained train set."""

    def __init__(self, dataset_size: int, samples_per_epoch: int | None, seed: int) -> None:
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive")
        self.dataset_size = int(dataset_size)
        self.samples_per_epoch = min(int(samples_per_epoch), dataset_size) if samples_per_epoch else dataset_size
        self.seed = int(seed)
        self.epoch = 1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        order = torch.randperm(self.dataset_size, generator=generator).tolist()
        start = ((self.epoch - 1) * self.samples_per_epoch) % self.dataset_size
        selected = [order[(start + offset) % self.dataset_size] for offset in range(self.samples_per_epoch)]
        batch_generator = torch.Generator().manual_seed(self.seed + 1009 * self.epoch)
        shuffled = torch.randperm(len(selected), generator=batch_generator).tolist()
        return iter(selected[index] for index in shuffled)
