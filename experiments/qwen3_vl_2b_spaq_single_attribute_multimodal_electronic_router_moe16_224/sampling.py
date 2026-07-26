from __future__ import annotations

import torch
from torch.utils.data import Sampler


class EpochRotatingSampler(Sampler[int]):
    """Deterministic shuffled epoch windows that eventually cover the retained train set."""

    def __init__(self, dataset_size: int, samples_per_epoch: int | None, seed: int,
                 shard_size: int | None = None, epoch_partitions: int | None = None) -> None:
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive")
        if samples_per_epoch is not None and epoch_partitions is not None:
            raise ValueError("samples_per_epoch and epoch_partitions are mutually exclusive")
        if epoch_partitions is not None and not 1 <= int(epoch_partitions) <= int(dataset_size):
            raise ValueError("epoch_partitions must be between 1 and dataset_size")
        self.dataset_size = int(dataset_size)
        self.samples_per_epoch = min(int(samples_per_epoch), dataset_size) if samples_per_epoch else dataset_size
        self.epoch_partitions = int(epoch_partitions) if epoch_partitions is not None else None
        self.seed = int(seed)
        self.shard_size = int(shard_size) if shard_size else None
        self.epoch = 1

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) <= 0:
            raise ValueError("epoch must be positive")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self.epoch_partitions is not None:
            return self._partition_sizes()[self.partition_index]
        return self.samples_per_epoch

    def __iter__(self):
        cycle = self.cycle_index if self.epoch_partitions is not None else 0
        generator = torch.Generator().manual_seed(self.seed + 104729 * cycle)
        order = self._dataset_order(generator)
        if self.epoch_partitions is not None:
            sizes = self._partition_sizes()
            start = sum(sizes[:self.partition_index])
            selected = order[start:start + sizes[self.partition_index]]
        else:
            start = ((self.epoch - 1) * self.samples_per_epoch) % self.dataset_size
            selected = [order[(start + offset) % self.dataset_size] for offset in range(self.samples_per_epoch)]
        batch_generator = torch.Generator().manual_seed(self.seed + 1009 * self.epoch)
        if self.shard_size is None:
            shuffled = torch.randperm(len(selected), generator=batch_generator).tolist()
            return iter(selected[index] for index in shuffled)

        # Shuffle shard order and samples within each shard, but keep a shard locally
        # contiguous. This preserves stochastic ordering without making the small LRU
        # cache reload multi-megabyte teacher/processor shards for almost every sample.
        groups: dict[int, list[int]] = {}
        for index in selected:
            groups.setdefault(index // self.shard_size, []).append(index)
        shard_numbers = list(groups)
        shard_order = torch.randperm(len(shard_numbers), generator=batch_generator).tolist()
        result: list[int] = []
        for offset in shard_order:
            group = groups[shard_numbers[offset]]
            order_in_shard = torch.randperm(len(group), generator=batch_generator).tolist()
            result.extend(group[position] for position in order_in_shard)
        return iter(result)

    def _dataset_order(self, generator: torch.Generator) -> list[int]:
        if self.epoch_partitions is None or self.shard_size is None:
            return torch.randperm(self.dataset_size, generator=generator).tolist()

        # A fully random sample partition touches almost every multi-megabyte
        # cache shard even when it contains only one third of the dataset.
        # Shuffle shard order and every shard's contents instead, then split
        # this equally random deterministic order into exact epoch partitions.
        # This preserves three-epoch coverage while loading roughly one third
        # of the teacher/processor shards per epoch.
        shards = [
            list(range(start, min(start + self.shard_size, self.dataset_size)))
            for start in range(0, self.dataset_size, self.shard_size)
        ]
        shard_order = torch.randperm(len(shards), generator=generator).tolist()
        order: list[int] = []
        for shard_index in shard_order:
            shard = shards[shard_index]
            within = torch.randperm(len(shard), generator=generator).tolist()
            order.extend(shard[index] for index in within)
        return order

    @property
    def partition_index(self) -> int:
        if self.epoch_partitions is None:
            return 0
        return (self.epoch - 1) % self.epoch_partitions

    @property
    def cycle_index(self) -> int:
        if self.epoch_partitions is None:
            return 0
        return (self.epoch - 1) // self.epoch_partitions

    def _partition_sizes(self) -> list[int]:
        if self.epoch_partitions is None:
            return [self.samples_per_epoch]
        base, remainder = divmod(self.dataset_size, self.epoch_partitions)
        return [
            base + (1 if index < remainder else 0)
            for index in range(self.epoch_partitions)
        ]

    def sampling_metadata(self) -> dict[str, int | str | list[int] | None]:
        return {
            "mode": "exact_epoch_partitions" if self.epoch_partitions is not None else "rotating_window",
            "dataset_size": self.dataset_size,
            "samples_this_epoch": len(self),
            "epoch_partitions": self.epoch_partitions,
            "partition_sizes": self._partition_sizes() if self.epoch_partitions is not None else None,
            "partition_index": self.partition_index if self.epoch_partitions is not None else None,
            "cycle_index": self.cycle_index if self.epoch_partitions is not None else None,
        }
