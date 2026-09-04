from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence
from typing import Any

import torch
from torch.utils.data import Sampler


class CyclicBalancedPKBatchSampler(Sampler[list[int]]):
    """Deterministic P=all-classes, K-image sampler with cross-epoch coverage.

    Each class owns an independently shuffled circular queue.  An image is not
    repeated within a class until every image in that class has been visited.
    Queue position is a pure function of ``seed`` and ``epoch``; restarting or
    running another architecture therefore produces exactly the same batches.

    Requiring every class in every batch makes the coverage contract explicit
    and also supplies all retrieval negatives in every optimizer step.
    """

    def __init__(
        self,
        samples: Sequence[Any],
        p: int,
        k: int,
        seed: int,
        steps_per_epoch: int | None = None,
    ) -> None:
        self.p = int(p)
        self.k = int(k)
        self.seed = int(seed)
        self.epoch = 0
        grouped: dict[int, list[int]] = defaultdict(list)
        for index, sample in enumerate(samples):
            grouped[int(sample.sku_index)].append(index)
        self.grouped = {
            sku: tuple(indexes) for sku, indexes in sorted(grouped.items())
        }
        if self.p != len(self.grouped):
            raise ValueError(
                "cyclic_balanced sampling requires P to equal the number of "
                f"training classes ({len(self.grouped)}), got P={self.p}"
            )
        if self.k <= 0 or any(not indexes for indexes in self.grouped.values()):
            raise ValueError("cyclic_balanced sampling requires K>0 and nonempty classes")
        if steps_per_epoch is None or int(steps_per_epoch) <= 0:
            raise ValueError(
                "cyclic_balanced sampling requires a positive "
                "batching.optimizer_steps_per_epoch"
            )
        self.batch_count = int(steps_per_epoch)
        self._permutation_cache: dict[tuple[int, int], tuple[int, ...]] = {}

    def set_epoch(self, epoch: int) -> None:
        # The shared trainer numbers epochs from one.  Store a zero-based
        # value so epoch 1 starts at queue offset zero.
        self.epoch = max(0, int(epoch) - 1)

    def __len__(self) -> int:
        return self.batch_count

    def _cycle_order(self, sku: int, cycle: int) -> tuple[int, ...]:
        key = (sku, cycle)
        cached = self._permutation_cache.get(key)
        if cached is not None:
            return cached
        indexes = self.grouped[sku]
        generator = torch.Generator().manual_seed(
            self.seed + 10_000_019 * (sku + 1) + 1_000_003 * cycle
        )
        order = torch.randperm(len(indexes), generator=generator).tolist()
        value = tuple(indexes[position] for position in order)
        self._permutation_cache[key] = value
        return value

    def _index_at(self, sku: int, absolute_position: int) -> int:
        size = len(self.grouped[sku])
        cycle, offset = divmod(int(absolute_position), size)
        return self._cycle_order(sku, cycle)[offset]

    def __iter__(self) -> Iterator[list[int]]:
        sku_ids = list(self.grouped)
        draws_per_class_per_epoch = self.batch_count * self.k
        epoch_start = self.epoch * draws_per_class_per_epoch
        for step in range(self.batch_count):
            batch: list[int] = []
            step_start = epoch_start + step * self.k
            # Shuffle whole class blocks, not individual samples.  Keeping the
            # K samples inside each class in queue order makes the
            # no-repeat-before-coverage contract exact even at a cycle/epoch
            # boundary, while still avoiding a fixed class-major order.
            generator = torch.Generator().manual_seed(
                self.seed + self.epoch * 1_000_003 + step
            )
            class_order = torch.randperm(
                len(sku_ids), generator=generator
            ).tolist()
            for class_position in class_order:
                sku = sku_ids[class_position]
                batch.extend(
                    self._index_at(sku, step_start + offset)
                    for offset in range(self.k)
                )
            yield batch


__all__ = ["CyclicBalancedPKBatchSampler"]
