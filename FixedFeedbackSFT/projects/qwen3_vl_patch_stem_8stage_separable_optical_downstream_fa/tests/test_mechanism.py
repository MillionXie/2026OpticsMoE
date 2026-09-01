from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from experiments.qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa.mechanism import (
    BLOCKS,
    UPDATE_METHODS,
    _phase_depth_rows,
    _swap_rows,
    build_parser,
    build_mechanism_plan,
    compose_parameter_state,
    partition_parameter_names,
    shapley_values,
)


def test_parser_accepts_locked_formal_repo_root() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--config",
            "formal.yaml",
            "--formal-repo-root",
            "/locked/p12",
        ]
    )

    assert args.config == Path("formal.yaml")
    assert args.formal_repo_root == Path("/locked/p12")


class _ToyStage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.raw_phase = nn.Parameter(torch.zeros(2, 2))
        self.electronic_skip = nn.Linear(2, 2)
        self.register_buffer(
            "feedback_phase", torch.full((2, 2), 17.0), persistent=False
        )


class _ToyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(2, 2)
        self.stem.requires_grad_(False)
        self.adapter = nn.Linear(2, 2)
        self.stages = nn.ModuleList([_ToyStage() for _ in range(8)])
        self.register_buffer("physical_transfer", torch.full((2, 2), 23.0))


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _ToyBackbone()
        self.head = nn.Linear(2, 3)
        self.register_buffer("source_phases", torch.full((8, 2, 2), 31.0))


def _constant_state(model: nn.Module, value: float) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if tensor.is_floating_point():
            result[name] = torch.full_like(tensor, value)
        else:
            result[name] = tensor.clone()
    return result


def test_parameter_partition_is_exact_and_excludes_only_frozen_stem() -> None:
    model = _ToyModel()
    partition = partition_parameter_names(model)

    assert len(partition.phase_by_stage) == 8
    assert all(len(stage) == 1 for stage in partition.phase_by_stage)
    assert set(partition.phase) == {
        f"backbone.stages.{index}.raw_phase" for index in range(8)
    }
    assert partition.electronics
    assert all(
        name.startswith("backbone.adapter.")
        or name.startswith("backbone.stages.")
        for name in partition.electronics
    )
    assert partition.head == ("head.bias", "head.weight")
    assert partition.frozen == (
        "backbone.stem.bias",
        "backbone.stem.weight",
    )
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert set(partition.all_parameters) == trainable


def test_composition_swaps_only_peh_parameters_and_retains_all_fixed_state() -> None:
    model = _ToyModel()
    partition = partition_parameter_names(model)
    template_state = {
        name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
    }
    donor_states = {
        "phase": _constant_state(model, 2.0),
        "electronics": _constant_state(model, 3.0),
        "head": _constant_state(model, 4.0),
    }

    composed, buffers = compose_parameter_state(
        model,
        partition,
        donor_states,
        phase_sources=("phase",) * 8,
        electronics_source="electronics",
        head_source="head",
    )

    for name in partition.phase:
        assert torch.equal(composed[name], donor_states["phase"][name])
    for name in partition.electronics:
        assert torch.equal(composed[name], donor_states["electronics"][name])
    for name in partition.head:
        assert torch.equal(composed[name], donor_states["head"][name])
    state_names = set(model.state_dict())
    parameter_names = {name for name, _ in model.named_parameters()}
    assert set(buffers) == state_names - parameter_names
    assert "backbone.stages.0.feedback_phase" not in composed
    for name in (*buffers, *partition.frozen):
        assert torch.equal(composed[name], template_state[name])
        assert not torch.equal(composed[name], donor_states["phase"][name])


def test_shapley_recovers_additive_block_effects_and_efficiency() -> None:
    effects = {"phase": 2.0, "electronics": -1.0, "head": 4.0}
    values = {
        frozenset(block for index, block in enumerate(BLOCKS) if mask & (1 << index)):
        5.0
        + sum(
            effects[block]
            for index, block in enumerate(BLOCKS)
            if mask & (1 << index)
        )
        for mask in range(8)
    }

    result = shapley_values(values)

    for block, expected in effects.items():
        assert math.isclose(result[block], expected, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(
        sum(result.values()),
        values[frozenset(BLOCKS)] - values[frozenset()],
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )


def test_mechanism_plan_has_complete_factorials_swaps_and_depth_resets() -> None:
    specs, links = build_mechanism_plan()

    assert len(specs) == 40
    assert len({spec.cache_key for spec in specs}) == len(specs)
    assert {spec.state_id for spec in specs} == {
        str(link["state_id"]) for link in links
    }
    for method in UPDATE_METHODS:
        factorial = [
            link
            for link in links
            if link["analysis"] == "factorial" and link["method"] == method
        ]
        assert len(factorial) == 8
        assert {frozenset(link["subset"]) for link in factorial} == {
            frozenset(block for index, block in enumerate(BLOCKS) if mask & (1 << index))
            for mask in range(8)
        }
        depth = [
            link
            for link in links
            if link["analysis"] == "phase_depth_reset" and link["method"] == method
        ]
        assert {link["kept"] for link in depth} == {
            "stage8_only",
            "stages1_to_7_only",
        }
        for analysis in ("phase_swap", "electronics_swap"):
            swaps = [
                link
                for link in links
                if link["analysis"] == analysis and link["recipient"] == method
            ]
            assert {link["donor"] for link in swaps} == {
                "common",
                *UPDATE_METHODS,
            }


def test_swap_and_phase_depth_summaries_use_directed_counterfactuals() -> None:
    specs, links = build_mechanism_plan()
    code = {"common": 0.0, "bp": 1.0, "fa_pretrained": 2.0, "fa_random": 3.0}
    metrics: dict[str, dict[str, float]] = {}
    for spec in specs:
        phase = sum(code[source] for source in spec.phase_sources) / 8.0
        value = phase + 2.0 * code[spec.electronics_source] + 3.0 * code[
            spec.head_source
        ]
        metrics[spec.state_id] = {"top1": value, "loss": 100.0 - value}

    swaps = _swap_rows(
        links,
        metrics,
        task="caltech101",
        seed=2026,
        endpoint="best",
        primary_metric="top1",
    )
    phase_transport = next(
        row
        for row in swaps
        if row["analysis"] == "phase_swap"
        and row["recipient"] == "bp"
        and row["donor"] == "fa_pretrained"
        and row["metric"] == "top1"
    )
    assert math.isclose(
        float(phase_transport["transport_ratio_to_recipient_own_effect"]), 2.0
    )
    electronics_transport = next(
        row
        for row in swaps
        if row["analysis"] == "electronics_swap"
        and row["recipient"] == "bp"
        and row["donor"] == "fa_pretrained"
        and row["metric"] == "top1"
    )
    assert math.isclose(
        float(electronics_transport["transport_ratio_to_recipient_own_effect"]),
        2.0,
    )

    depth = _phase_depth_rows(
        links,
        metrics,
        task="caltech101",
        seed=2026,
        endpoint="best",
        primary_metric="top1",
    )
    bp_depth = next(
        row
        for row in depth
        if row["method"] == "bp" and row["metric"] == "top1"
    )
    assert math.isclose(float(bp_depth["stages1_to_7_recovery_ratio"]), 7.0 / 8.0)
    assert math.isclose(float(bp_depth["stage8_recovery_ratio"]), 1.0 / 8.0)
    assert math.isclose(float(bp_depth["early_late_interaction"]), 0.0, abs_tol=1e-12)
