from __future__ import annotations

import random
from collections import Counter
from typing import Iterator, Sequence

from torch.utils.data import Sampler


class EpochClassMixedSampler(Sampler[int]):
    """Rotate a per-class epoch window, then interleave and shuffle classes per batch."""

    def __init__(
        self,
        indices: Sequence[int],
        labels: Sequence[int],
        num_classes: int,
        batch_size: int,
        seed: int,
        per_class_limit: int | None = None,
        shard_size: int | None = None,
        oversample_minority: bool = False,
    ) -> None:
        self.indices = [int(index) for index in indices]
        self.labels = [int(label) for label in labels]
        self.num_classes = int(num_classes)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.per_class_limit = int(per_class_limit) if per_class_limit is not None else None
        self.shard_size = int(shard_size) if shard_size is not None else None
        self.oversample_minority = bool(oversample_minority)
        self.epoch = 1
        if self.batch_size <= 0 or self.num_classes <= 0:
            raise ValueError("batch_size and num_classes must be positive")
        if self.per_class_limit is not None and self.per_class_limit <= 0:
            raise ValueError("per_class_limit must be positive when set")
        if self.shard_size is not None and self.shard_size <= 0:
            raise ValueError("shard_size must be positive when set")
        if any(index < 0 or index >= len(self.labels) for index in self.indices):
            raise ValueError("sampler indices do not match labels")
        self._by_class = {
            class_index: [index for index in self.indices if self.labels[index] == class_index]
            for class_index in range(self.num_classes)
        }
        missing = [class_index for class_index, values in self._by_class.items() if not values]
        if missing:
            raise ValueError(f"No training samples for classes: {missing}")

    def __len__(self) -> int:
        return sum(
            self.per_class_limit
            if self.per_class_limit is not None and self.oversample_minority
            else min(len(values), self.per_class_limit)
            if self.per_class_limit is not None
            else len(values)
            for values in self._by_class.values()
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def epoch_class_counts(self) -> dict[int, int]:
        return {
            class_index: self.per_class_limit
            if self.per_class_limit is not None and self.oversample_minority
            else min(len(values), self.per_class_limit)
            if self.per_class_limit is not None
            else len(values)
            for class_index, values in self._by_class.items()
        }

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + 1_000_003 * self.epoch)
        queues = {
            class_index: self._ordered_epoch_indices(class_index, rng)
            for class_index in range(self.num_classes)
        }
        positions = {class_index: 0 for class_index in range(self.num_classes)}
        remaining = sum(len(values) for values in queues.values())
        while remaining:
            batch: list[int] = []
            while len(batch) < self.batch_size and remaining:
                available = [
                    class_index
                    for class_index in range(self.num_classes)
                    if positions[class_index] < len(queues[class_index])
                ]
                rng.shuffle(available)
                progressed = False
                for class_index in available:
                    if len(batch) >= self.batch_size:
                        break
                    position = positions[class_index]
                    if position < len(queues[class_index]):
                        batch.append(queues[class_index][position])
                        positions[class_index] = position + 1
                        remaining -= 1
                        progressed = True
                if not progressed:
                    raise RuntimeError("Class-mixed sampler could not make progress")
            yield from batch

    def _ordered_epoch_indices(self, class_index: int, rng: random.Random) -> list[int]:
        source = list(self._by_class[class_index])
        fixed = random.Random(self.seed + 97_409 * (class_index + 1))
        fixed.shuffle(source)
        if self.per_class_limit is not None and self.per_class_limit < len(source):
            start = ((self.epoch - 1) * self.per_class_limit) % len(source)
            selected = [source[(start + offset) % len(source)] for offset in range(self.per_class_limit)]
        elif self.per_class_limit is not None and self.oversample_minority and self.per_class_limit > len(source):
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
        ordered: list[int] = []
        for shard_number in shard_numbers:
            values = groups[shard_number]
            rng.shuffle(values)
            ordered.extend(values)
        return ordered


def batch_class_counts(indices: Sequence[int], labels: Sequence[int]) -> dict[int, int]:
    return dict(Counter(int(labels[int(index)]) for index in indices))

