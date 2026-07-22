from __future__ import annotations

import torch
from torch.utils.data import Sampler


class EpochRotatingSampler(Sampler[int]):
    """Deterministic shuffled epoch windows that eventually cover the retained train set."""

    def __init__(self, dataset_size: int, samples_per_epoch: int | None, seed: int,
                 shard_size: int | None = None) -> None:
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive")
        self.dataset_size = int(dataset_size)
        self.samples_per_epoch = min(int(samples_per_epoch), dataset_size) if samples_per_epoch else dataset_size
        self.seed = int(seed)
        self.shard_size = int(shard_size) if shard_size else None
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


class BalancedEpochRotatingSampler(Sampler[int]):
    """Exact per-class epoch windows with deterministic full-dataset rotation.

    Selected indices remain locally contiguous by cache shard. Shard order and
    sample order inside every shard change deterministically each epoch.
    """

    def __init__(self, labels: list[int] | tuple[int, ...], samples_per_class: int,
                 seed: int, shard_size: int | None = None) -> None:
        if samples_per_class <= 0:
            raise ValueError("samples_per_class must be positive")
        self.labels = [int(label) for label in labels]
        self.samples_per_class = int(samples_per_class)
        self.seed = int(seed)
        self.shard_size = int(shard_size) if shard_size else None
        if self.shard_size is not None and self.shard_size <= 0:
            raise ValueError("shard_size must be positive when set")
        self.classes = sorted(set(self.labels))
        if len(self.classes) < 2:
            raise ValueError("Balanced sampling requires at least two classes")
        self.by_class = {
            label: [index for index, value in enumerate(self.labels) if value == label]
            for label in self.classes
        }
        too_small = {label: len(indices) for label, indices in self.by_class.items()
                     if len(indices) < self.samples_per_class}
        if too_small:
            raise ValueError(
                f"samples_per_class={self.samples_per_class} exceeds available samples: {too_small}"
            )
        self.fixed_orders = {
            label: self._fixed_order(label, indices) for label, indices in self.by_class.items()
        }
        self.epoch = 1

    def _fixed_order(self, label: int, indices: list[int]) -> list[int]:
        generator = torch.Generator().manual_seed(self.seed + 97_409 * (label + 1))
        permutation = torch.randperm(len(indices), generator=generator).tolist()
        return [indices[position] for position in permutation]

    def set_epoch(self, epoch: int) -> None:
        if epoch <= 0:
            raise ValueError("epoch must be positive")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.classes) * self.samples_per_class

    def epoch_class_counts(self) -> dict[int, int]:
        return {label: self.samples_per_class for label in self.classes}

    def __iter__(self):
        selected: list[int] = []
        for label in self.classes:
            source = self.fixed_orders[label]
            start = ((self.epoch - 1) * self.samples_per_class) % len(source)
            selected.extend(source[(start + offset) % len(source)]
                            for offset in range(self.samples_per_class))

        generator = torch.Generator().manual_seed(self.seed + 1_000_003 * self.epoch)
        if self.shard_size is None:
            order = torch.randperm(len(selected), generator=generator).tolist()
            return iter(selected[position] for position in order)

        groups: dict[int, list[int]] = {}
        for index in selected:
            groups.setdefault(index // self.shard_size, []).append(index)
        shard_numbers = list(groups)
        shard_order = torch.randperm(len(shard_numbers), generator=generator).tolist()
        result: list[int] = []
        for position in shard_order:
            group = groups[shard_numbers[position]]
            within = torch.randperm(len(group), generator=generator).tolist()
            result.extend(group[index] for index in within)
        return iter(result)
