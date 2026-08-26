"""Pure selection contracts for demonstration and unbiased hardware profiles."""

from __future__ import annotations

from typing import Any

import numpy as np


PHASE_FILENAME = "mnist4_single_layer_17um_10cm.bmp"
DEMO_PROFILE = "demo_topk"


def formal_profile_name(samples_per_class: int) -> str:
    count = int(samples_per_class)
    if count <= 0:
        raise ValueError("samples_per_class must be positive")
    return f"formal_fixed_random_{count}_per_class"


def select_demo_topk(
    candidates: dict[int, list[dict[str, Any]]],
    classes: tuple[int, ...],
    samples_per_class: int,
) -> dict[int, list[dict[str, Any]]]:
    """Select visually clear demonstrations; never use this set as a test score."""

    result: dict[int, list[dict[str, Any]]] = {}
    for label in classes:
        ranked = sorted(
            candidates[label],
            key=lambda item: (item["correct"], item["margin"]),
            reverse=True,
        )
        result[label] = ranked[: min(int(samples_per_class), len(ranked))]
    return result


def select_formal_fixed_random(
    candidates: dict[int, list[dict[str, Any]]],
    classes: tuple[int, ...],
    *,
    samples_per_class: int,
    seed: int,
    require_full_count: bool,
) -> dict[int, list[dict[str, Any]]]:
    """Select by label and a fixed RNG only, never by model output or correctness."""

    generator = np.random.default_rng(int(seed))
    result: dict[int, list[dict[str, Any]]] = {}
    for label in classes:
        ordered = sorted(candidates[label], key=lambda item: item["dataset_index"])
        if require_full_count and len(ordered) < int(samples_per_class):
            raise RuntimeError(
                f"Formal export needs {samples_per_class} class-{label} samples, "
                f"but the dataset contains only {len(ordered)}"
            )
        count = min(int(samples_per_class), len(ordered))
        indices = generator.choice(len(ordered), size=count, replace=False)
        result[label] = [ordered[int(index)] for index in indices]
    return result
