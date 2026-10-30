"""Explicit latest human DEV comparison tolerance; not an attribute evaluator."""
import math


def metric_comparison(actual,reference,tolerance=.10):
    if not all(math.isfinite(v) for v in [actual,reference,tolerance]) or not 0<=tolerance<1:
        raise ValueError('Finite values and explicit relative tolerance required')
    if abs(reference)<1e-8:
        raise ValueError('A percentage relative to a zero score is undefined; report raw differences')
    # ImageReward may be negative. Dividing actual/reference would reverse
    # preference ordering, so use the signed change over |baseline| instead.
    drop=(reference-actual)/abs(reference)
    return dict(actual=actual,reference=reference,signed_drop_over_abs_reference=drop,
        max_drop=tolerance,passed=drop<=tolerance,
        negative_reference_normalization=reference<0,
        normalized_preference_change_is_not_probability=True)
