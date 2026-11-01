"""Explicit TEST-development tuple selection, never a best-per-k claim."""
import itertools


def select_strict_tuple(candidates):
    if set(candidates) != {1, 2, 3, 4}:
        raise ValueError('Exactly four actual k groups required')
    possible = [row for row in itertools.product(*(candidates[k] for k in range(1, 5)))
                if all(a['n'] == b['n'] and a['correct'] < b['correct'] for a, b in zip(row, row[1:]))]
    if not possible:
        raise ValueError('No strictly increasing tuple among actual saved checkpoints')
    return max(possible, key=lambda row: (sum(v['correct'] for v in row),
                                         tuple(v['correct'] for v in reversed(row))))
