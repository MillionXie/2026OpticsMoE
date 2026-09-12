"""Explicit training RNG streams; no model, CUDA, data split or inference changes."""
import random


def training_seed(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError('Training seed must be an integer in [0, 2**32-1]')
    seed = int(value)
    if not 0 <= seed < 2**32:
        raise ValueError('Training seed must be an integer in [0, 2**32-1]')
    return seed


def epoch_random_streams(seed, epoch):
    """Seed42 exactly preserves legacy 42+epoch / 19042+epoch streams."""
    seed = training_seed(seed)
    if type(epoch) is not int or epoch < 1:
        raise ValueError('Positive integer training epoch required')
    return random.Random(seed + epoch), random.Random(seed + 19000 + epoch)
