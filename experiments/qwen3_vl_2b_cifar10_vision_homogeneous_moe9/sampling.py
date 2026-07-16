from __future__ import annotations

import random
from typing import Iterator, Sequence

from torch.utils.data import Sampler


class EpochClassMixedSampler(Sampler[int]):
    """Balanced batches with rotating per-class windows across epochs."""

    def __init__(self, indices: Sequence[int], labels: Sequence[int], num_classes: int, batch_size: int,
                 seed: int, per_class_limit: int | None = None, shard_size: int | None = None) -> None:
        self.indices = [int(value) for value in indices]
        self.labels = [int(value) for value in labels]
        self.num_classes = int(num_classes)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.per_class_limit = int(per_class_limit) if per_class_limit is not None else None
        self.shard_size = int(shard_size) if shard_size else None
        self.epoch = 1
        self._by_class = {class_index: [index for index in self.indices if self.labels[index] == class_index]
                          for class_index in range(self.num_classes)}
        missing = [key for key, values in self._by_class.items() if not values]
        if missing:
            raise ValueError(f"No training samples for classes: {missing}")

    def __len__(self) -> int:
        return sum(min(len(values), self.per_class_limit) if self.per_class_limit is not None else len(values)
                   for values in self._by_class.values())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def epoch_class_counts(self) -> dict[int, int]:
        return {key: min(len(values), self.per_class_limit) if self.per_class_limit is not None else len(values)
                for key, values in self._by_class.items()}

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + 1_000_003 * self.epoch)
        queues = {class_index: self._epoch_indices(class_index, rng) for class_index in range(self.num_classes)}
        positions = {class_index: 0 for class_index in range(self.num_classes)}
        remaining = sum(len(values) for values in queues.values())
        while remaining:
            batch: list[int] = []
            while len(batch) < self.batch_size and remaining:
                available = [key for key in range(self.num_classes) if positions[key] < len(queues[key])]
                rng.shuffle(available)
                for class_index in available:
                    if len(batch) >= self.batch_size:
                        break
                    batch.append(queues[class_index][positions[class_index]])
                    positions[class_index] += 1
                    remaining -= 1
            yield from batch

    def _epoch_indices(self, class_index: int, rng: random.Random) -> list[int]:
        source = list(self._by_class[class_index])
        fixed = random.Random(self.seed + 97_409 * (class_index + 1))
        fixed.shuffle(source)
        if self.per_class_limit is not None and self.per_class_limit < len(source):
            start = ((self.epoch - 1) * self.per_class_limit) % len(source)
            selected = [source[(start + offset) % len(source)] for offset in range(self.per_class_limit)]
        else:
            selected = source
        if self.shard_size is None:
            rng.shuffle(selected)
            return selected
        groups: dict[int, list[int]] = {}
        for index in selected:
            groups.setdefault(index // self.shard_size, []).append(index)
        shard_numbers = list(groups)
        rng.shuffle(shard_numbers)
        result: list[int] = []
        for number in shard_numbers:
            rng.shuffle(groups[number])
            result.extend(groups[number])
        return result
