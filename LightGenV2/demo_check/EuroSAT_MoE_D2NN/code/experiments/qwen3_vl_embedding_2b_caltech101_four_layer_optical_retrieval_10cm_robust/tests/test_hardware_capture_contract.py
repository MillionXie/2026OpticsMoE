from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ..hardware_bridge import (
    STAGES,
    _capture_contract_parameters,
    _downstream_parameters,
    _split_hardware_development,
)


def _linear() -> nn.Module:
    return nn.Linear(2, 2)


class _FakePhysicalCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_adapter = _linear()
        self.input_norm = nn.LayerNorm(2)
        self.router = _linear()
        self.expert_layers = nn.ModuleList([_linear()])
        self.global_phase = _linear()
        self.readout = _linear()
        self.output_adapter = _linear()


class _FakeOpticalBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.core = _FakePhysicalCore()
        self.expert_readout = _linear()
        self.expert_output_adapter = _linear()


class _FakeCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_adapter = _linear()
        self.input_norm = nn.LayerNorm(2)
        self.blocks = nn.ModuleList([_linear(), _linear()])
        self.output_norm = nn.LayerNorm(2)
        self.output_adapter = _linear()
        self.optical_branch = _FakeOpticalBranch()
        self.block1_optical_fusion_logit = nn.Parameter(torch.zeros(()))
        self.block2_optical_fusion_logit = nn.Parameter(torch.zeros(()))
        self.residual_logit = nn.Parameter(torch.zeros(()))


class _FakeSurrogate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.core = _FakeCore()


def _fixture() -> tuple[SimpleNamespace, nn.Module]:
    replacement = SimpleNamespace(
        vision_surrogate=_FakeSurrogate(),
        language_surrogate=_FakeSurrogate(),
        vision_pre_attention=None,
        language_pre_attention=None,
    )
    return replacement, _linear()


@pytest.mark.parametrize("stage", STAGES)
def test_downstream_policy_never_trains_captured_upstream(stage: str) -> None:
    replacement, readout = _fixture()
    parameters = _downstream_parameters(replacement, readout, stage)
    contract = _capture_contract_parameters(replacement, stage)
    assert contract
    assert all(not parameter.requires_grad for parameter in contract.values())
    assert {id(parameter) for parameter in parameters} == {
        id(parameter)
        for module in (
            replacement.vision_surrogate,
            replacement.language_surrogate,
            readout,
        )
        for parameter in module.parameters()
        if parameter.requires_grad
    }


def test_stage_specific_capture_boundaries_are_strict() -> None:
    replacement, readout = _fixture()
    v = replacement.vision_surrogate.core

    _downstream_parameters(replacement, readout, "vision_expert")
    assert not any(parameter.requires_grad for parameter in v.input_adapter.parameters())
    assert all(parameter.requires_grad for parameter in v.blocks.parameters())
    assert not any(
        parameter.requires_grad
        for parameter in v.optical_branch.core.expert_layers.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in v.optical_branch.core.global_phase.parameters()
    )

    replacement, readout = _fixture()
    v = replacement.vision_surrogate.core
    _downstream_parameters(replacement, readout, "vision_global")
    assert not any(parameter.requires_grad for parameter in v.blocks[0].parameters())
    assert all(parameter.requires_grad for parameter in v.blocks[1].parameters())
    assert not any(
        parameter.requires_grad
        for parameter in v.optical_branch.core.global_phase.parameters()
    )

    replacement, readout = _fixture()
    l = replacement.language_surrogate.core
    _downstream_parameters(replacement, readout, "language_expert")
    assert not any(parameter.requires_grad for parameter in l.input_adapter.parameters())
    assert not any(parameter.requires_grad for parameter in l.input_norm.parameters())
    assert all(parameter.requires_grad for parameter in l.blocks.parameters())
    assert not any(
        parameter.requires_grad
        for parameter in l.optical_branch.core.expert_layers.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in l.optical_branch.core.global_phase.parameters()
    )

    replacement, readout = _fixture()
    l = replacement.language_surrogate.core
    _downstream_parameters(replacement, readout, "language_global")
    assert not any(parameter.requires_grad for parameter in l.blocks[0].parameters())
    assert all(parameter.requires_grad for parameter in l.blocks[1].parameters())
    assert not any(
        parameter.requires_grad
        for parameter in l.optical_branch.core.global_phase.parameters()
    )


def test_development_split_is_fixed_balanced_and_disjoint() -> None:
    grouped = {
        label: [SimpleNamespace(sku_index=label, sample_id=f"{label}:{index}") for index in range(10)]
        for label in range(10)
    }
    fitting, support, query = _split_hardware_development(
        grouped, seed=42, development_per_class=2
    )
    repeated = _split_hardware_development(
        grouped, seed=42, development_per_class=2
    )
    assert [[item.sample_id for item in fitting[label]] for label in fitting] == [
        [item.sample_id for item in repeated[0][label]] for label in repeated[0]
    ]
    assert all(len(values) == 8 for values in fitting.values())
    assert len(support) == len(query) == 10
    fitting_ids = {item.sample_id for values in fitting.values() for item in values}
    development_ids = {item.sample_id for item in support + query}
    assert fitting_ids.isdisjoint(development_ids)
