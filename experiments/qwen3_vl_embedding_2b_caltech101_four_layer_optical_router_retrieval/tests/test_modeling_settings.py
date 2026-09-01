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
    router_contract_sha256,
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
    assert all(value.weight_decay == 0.0 for value in settings)
    assert all(value.random_seed == 42 for value in settings)
    assert all(value.router_optimization_seed == 42 for value in settings)
    optical = settings[-1]
    assert optical.optical_router_detector_intervals == ((164, 223), (255, 314))
    assert optical.optical_router_required_center_angle_deg < 0.65
    assert len(optical.router_contract_sha256) == 64
    assert optical.router_contract["optical"]["detector_intervals"] == [
        [164, 223],
        [255, 314],
    ]


def test_measurement_quality_gates_do_not_change_checkpoint_architecture() -> None:
    settings = load_settings(CONFIG_ROOT / "optical_power_topk2.yaml")
    original_hash = settings.router_contract_sha256
    original_label = architecture_label(settings)

    settings.optical_router_maximum_saturated_pixel_fraction = 0.75
    settings.optical_router_minimum_p99_uint8 = 200.0
    settings.optical_router_minimum_dynamic_range_uint8 = 100.0
    settings.optical_router_minimum_topk_probability_margin = 0.20

    recomputed_hash = router_contract_sha256(settings)
    assert recomputed_hash == original_hash
    settings.router_contract_sha256 = recomputed_hash
    assert architecture_label(settings) == original_label
    assert "hardware_quality" not in settings.router_contract["optical"]


def test_strict_electronic_transplant_loads_every_tensor() -> None:
    target = {"body": torch.zeros(2), f"{ROUTER_PREFIX}router.gate.weight": torch.zeros(1)}
    source = {"body": torch.ones(2), f"{ROUTER_PREFIX}router.gate.weight": torch.ones(1)}
    loaded, report = _copy_surrogate_state(target, source, reset_router=False)
    assert torch.equal(loaded["body"], torch.ones(2))
    assert torch.equal(loaded[f"{ROUTER_PREFIX}router.gate.weight"], torch.ones(1))
    assert report["new_router_tensor_count"] == 0


def test_repeated_seed_configs_keep_dataset_and_architecture_paired() -> None:
    for stem in (
        "electronic_power_topk1",
        "electronic_power_topk2",
        "electronic_power_topk4",
        "optical_power_topk2",
    ):
        values = [
            load_settings(CONFIG_ROOT / f"{stem}.yaml"),
            load_settings(CONFIG_ROOT / f"{stem}_seed43.yaml"),
            load_settings(CONFIG_ROOT / f"{stem}_seed44.yaml"),
        ]
        assert [item.router_optimization_seed for item in values] == [42, 43, 44]
        assert {item.random_seed for item in values} == {42}
        assert len({architecture_label(item) for item in values}) == 1
        assert len({item.router_contract_sha256 for item in values}) == 1
        assert len({item.output_dir for item in values}) == 3


def test_fair_reset_transplant_keeps_new_router_and_loads_common_body() -> None:
    target = {"body": torch.zeros(2), f"{ROUTER_PREFIX}raw_router_phase": torch.full((2,), 7.0)}
    source = {"body": torch.ones(2), f"{ROUTER_PREFIX}router.gate.weight": torch.full((1,), 9.0)}
    loaded, report = _copy_surrogate_state(target, source, reset_router=True)
    assert torch.equal(loaded["body"], torch.ones(2))
    assert torch.equal(loaded[f"{ROUTER_PREFIX}raw_router_phase"], torch.full((2,), 7.0))
    assert report["new_router_tensor_count"] == 1
    assert report["discarded_source_router_tensor_count"] == 1
