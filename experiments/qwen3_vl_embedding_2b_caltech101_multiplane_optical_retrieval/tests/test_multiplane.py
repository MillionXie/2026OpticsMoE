from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from experiments.qwen3_vl_embedding_2b_caltech101_electronic_retrieval.electronic_blocks import (
    ElectronicResidualMLPBlock,
)

from ..optical_cores import D2NNFivePlaneCore, MultiplaneMoECore, build_physical_core
from ..sampling import CyclicBalancedPKBatchSampler
from ..settings import load_settings


ROOT = Path(__file__).resolve().parents[1]


def _settings(variant: str):
    settings = load_settings(ROOT / "configs" / "release" / f"{variant}.yaml")
    settings.k_space_constraint_enabled = False
    settings.phase_dropout_mode = "none"
    settings.phase_dropout_p = 0.0
    return settings


def _phase_parameters(core: nn.Module) -> list[nn.Parameter]:
    return [
        module.raw_phase
        for module in core.modules()
        if hasattr(module, "raw_phase")
        and isinstance(module.raw_phase, nn.Parameter)
    ]


def test_continuous_moe_has_four_expert_planes_one_global_and_one_router() -> None:
    core = build_physical_core(16, 224, _settings("moe_continuous_fixed_router"))
    assert isinstance(core, MultiplaneMoECore)
    assert len(core.expert_layers) == 4
    assert len(core.additional_routers) == 0
    assert len(core.oeo_layers) == 0
    assert sum(len(layer.experts) for layer in core.expert_layers) == 16
    assert len(_phase_parameters(core)) == 17


def test_d2nn_is_five_224_planes_without_router_or_oeo() -> None:
    core = build_physical_core(16, 224, _settings("d2nn_continuous"))
    assert isinstance(core, D2NNFivePlaneCore)
    assert len(core.expert_layers) == 5
    assert all(layer.raw_phase.shape == (224, 224) for layer in core.expert_layers)
    assert core.all_router_parameters() == []
    assert len(_phase_parameters(core)) == 5


def test_d2nn_oeo_has_four_full_aperture_sigmoid_reload_boundaries() -> None:
    core = build_physical_core(8, 224, _settings("d2nn_oeo_sigmoid"))
    output = core.forward_groups([torch.randn(3, 8), torch.randn(2, 8)])
    assert isinstance(core, D2NNFivePlaneCore)
    assert output.shape == (5, 8)
    assert len(core.expert_layers) == 5
    assert len(core.oeo_layers) == 4
    assert core.all_router_parameters() == []
    assert len(core.stage_diagnostics) == 5
    assert all("reload_power" in values for values in core.stage_diagnostics[:4])
    assert "reload_power" not in core.stage_diagnostics[4]
    for conversion in core.oeo_layers:
        assert conversion.last_amplitude is not None
        assert torch.all(
            (conversion.last_amplitude >= 0)
            & (conversion.last_amplitude <= 1)
        )


def test_fixed_oeo_reuses_one_router_and_sigmoid_reload_is_nonnegative() -> None:
    core = build_physical_core(8, 224, _settings("moe_oeo_fixed_router"))
    output = core.forward_groups([torch.randn(3, 8)])
    assert output.shape == (3, 8)
    assert len(core.oeo_layers) == 4
    assert len(core.stage_routings) == 1
    for conversion in core.oeo_layers:
        assert conversion.last_amplitude is not None
        assert torch.all((conversion.last_amplitude >= 0) & (conversion.last_amplitude <= 1))


def test_dynamic_oeo_has_four_independent_router_calls() -> None:
    core = build_physical_core(8, 224, _settings("moe_oeo_dynamic_router"))
    output = core.forward_groups([torch.randn(3, 8), torch.randn(2, 8)])
    assert output.shape == (5, 8)
    assert len(core.additional_routers) == 3
    assert len(core.stage_routings) == 4
    assert len({id(module) for module in [core.router, *core.additional_routers]}) == 4


def test_every_phase_receives_a_finite_gradient() -> None:
    for variant in (
        "moe_continuous_fixed_router",
        "d2nn_continuous",
        "d2nn_oeo_sigmoid",
        "moe_oeo_fixed_router",
        "moe_oeo_dynamic_router",
    ):
        core = build_physical_core(8, 224, _settings(variant))
        output = core.forward_groups([torch.randn(3, 8)])
        output.square().mean().backward()
        phases = _phase_parameters(core)
        assert phases
        assert all(parameter.grad is not None for parameter in phases), variant
        assert all(torch.isfinite(parameter.grad).all() for parameter in phases), variant


def test_optical_cores_do_not_contain_electronic_residual_mixers() -> None:
    for variant in (
        "moe_continuous_fixed_router",
        "d2nn_continuous",
        "d2nn_oeo_sigmoid",
        "moe_oeo_fixed_router",
        "moe_oeo_dynamic_router",
    ):
        core = build_physical_core(8, 224, _settings(variant))
        assert not any(isinstance(module, ElectronicResidualMLPBlock) for module in core.modules())
        assert not any("fusion_logit" in name or "residual_logit" in name for name, _ in core.named_parameters())


def test_cyclic_sampler_covers_each_class_before_repeating() -> None:
    class Sample:
        def __init__(self, sku_index: int) -> None:
            self.sku_index = sku_index

    samples = [Sample(sku) for sku in range(3) for _ in range(7)]
    sampler = CyclicBalancedPKBatchSampler(
        samples, p=3, k=2, seed=42, steps_per_epoch=2
    )
    seen: dict[int, list[int]] = {sku: [] for sku in range(3)}
    for epoch in (1, 2):
        sampler.set_epoch(epoch)
        for batch in sampler:
            for index in batch:
                seen[samples[index].sku_index].append(index)
    for indexes in seen.values():
        assert len(indexes) == 8
        assert len(set(indexes[:7])) == 7
