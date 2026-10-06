"""Deterministic support sampling and class-disjoint transfer contracts."""
import random


def validate_classes(source, novel_sets):
    seen = set(source)
    if len(seen) != 10:
        raise ValueError('Expected ten source classes')
    for name, classes in novel_sets.items():
        if len(classes) != 10 or len(set(classes)) != 10 or seen.intersection(classes):
            raise ValueError(f'Invalid or overlapping novel classes: {name}')
        if any(c < 0 or c >= 100 for c in classes):
            raise ValueError('CIFAR fine class outside range')
        seen.update(classes)


def select_support(samples, shots, seed):
    """Nested 5/20-shot draws; no test examples, fixed order across methods."""
    if shots < 3:
        raise ValueError('PK training requires at least three examples per class')
    support = []
    for label in sorted({s.sku_index for s in samples}):
        rows = sorted((s for s in samples if s.sku_index == label), key=lambda s: s.sample_id)
        if any(s.source_split != 'official_train' for s in rows) or len(rows) < shots:
            raise ValueError('Support must come exclusively from sufficient official train examples')
        random.Random(f'{seed}:novel-support:{rows[0].sku_id}').shuffle(rows)
        support.extend(rows[:shots])
    return support
