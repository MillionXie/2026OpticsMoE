from __future__ import annotations

from pathlib import Path

import torch

from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval.modeling import (
    ROUTER_PREFIX,
    _copy_surrogate_state,
    architecture_label,
)
from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval.settings import (
    load_settings,
)


CONFIG_ROOT = Path(__file__).parents[1] / "configs" / "release"


def test_release_configs_have_distinct_strict_architectures() -> None:
    names = (
        "electronic_legacy_topk2_anchor.yaml",
        "electronic_power_topk1.yaml",
        "electronic_power_topk2.yaml",
        "electronic_power_topk4.yaml",
        "optical_power_topk2.yaml",
    )
    settings = [load_settings(CONFIG_ROOT / name) for name in names]
    labels = [architecture_label(value) for value in settings]
    assert len(set(labels)) == len(labels)
    assert settings[0].router_reset_parameters is False
    assert settings[0].router_straight_through is False
    assert all(value.router_reset_parameters for value in settings[1:])
    assert all(value.router_straight_through for value in settings[1:])
    assert all(value.phase_focus_enabled is False for value in settings)


def test_strict_electronic_transplant_loads_every_tensor() -> None:
    target = {"body": torch.zeros(2), f"{ROUTER_PREFIX}router.gate.weight": torch.zeros(1)}
    source = {"body": torch.ones(2), f"{ROUTER_PREFIX}router.gate.weight": torch.ones(1)}
    loaded, report = _copy_surrogate_state(target, source, reset_router=False)
    assert torch.equal(loaded["body"], torch.ones(2))
    assert torch.equal(loaded[f"{ROUTER_PREFIX}router.gate.weight"], torch.ones(1))
    assert report["new_router_tensor_count"] == 0


def test_fair_reset_transplant_keeps_new_router_and_loads_common_body() -> None:
    target = {"body": torch.zeros(2), f"{ROUTER_PREFIX}raw_router_phase": torch.full((2,), 7.0)}
    source = {"body": torch.ones(2), f"{ROUTER_PREFIX}router.gate.weight": torch.full((1,), 9.0)}
    loaded, report = _copy_surrogate_state(target, source, reset_router=True)
    assert torch.equal(loaded["body"], torch.ones(2))
    assert torch.equal(loaded[f"{ROUTER_PREFIX}raw_router_phase"], torch.full((2,), 7.0))
    assert report["new_router_tensor_count"] == 1
    assert report["discarded_source_router_tensor_count"] == 1
