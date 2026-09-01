from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from . import settings as settings_module
from .queue import completion_reason
from .settings import TASKS, Settings, implementation_sha256, load_settings
from .training import (
    CHECKPOINT_FORMAT,
    _validate_checkpoint_identity,
    build_data,
    build_model,
    configure_runtime_feedback,
    evaluate,
    module_state_sha256,
    seed_everything,
    sha256_file,
    sha256_json,
    write_json,
)


MECHANISM_FORMAT = "p12-fixed-feedback-mechanism-v1"
UPDATE_METHODS = ("bp", "fa_pretrained", "fa_random")
BLOCKS = ("phase", "electronics", "head")
_PHASE_NAME = re.compile(r"^backbone\.stages\.(\d+)\.raw_phase$")
_P11_SIGNATURE_KEY = "backbone.p11_separable_architecture_signature"
_P11_SIGNATURE = torch.tensor([11, 1, 2, 4], dtype=torch.int64)


@dataclass(frozen=True)
class ParameterPartition:
    phase_by_stage: tuple[tuple[str, ...], ...]
    electronics: tuple[str, ...]
    head: tuple[str, ...]
    frozen: tuple[str, ...]

    @property
    def phase(self) -> tuple[str, ...]:
        return tuple(name for stage in self.phase_by_stage for name in stage)

    @property
    def all_parameters(self) -> tuple[str, ...]:
        return self.phase + self.electronics + self.head


@dataclass(frozen=True)
class HybridSpec:
    state_id: str
    phase_sources: tuple[str, ...]
    electronics_source: str
    head_source: str

    @property
    def cache_key(self) -> tuple[Any, ...]:
        return (self.phase_sources, self.electronics_source, self.head_source)


def partition_parameter_names(model: nn.Module) -> ParameterPartition:
    """Partition every trainable tensor into optical phase, electronic body or head.

    Persistent buffers (stem weights, propagation transfer functions, random
    ablation phases, architecture signature and source phases) are deliberately
    absent. Mechanism hybrids always retain buffers from a fresh source model.
    """

    phase: dict[int, list[str]] = {}
    electronics: list[str] = []
    head: list[str] = []
    unknown: list[str] = []
    frozen: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            if not name.startswith("backbone.stem."):
                raise RuntimeError(
                    f"Unexpected frozen parameter outside the Qwen stem: {name!r}"
                )
            frozen.append(name)
            continue
        match = _PHASE_NAME.fullmatch(name)
        if match is not None:
            phase.setdefault(int(match.group(1)), []).append(name)
        elif name.startswith("head."):
            head.append(name)
        elif name.startswith("backbone.adapter.") or name.startswith(
            "backbone.stages."
        ):
            electronics.append(name)
        else:
            unknown.append(name)
    if unknown:
        raise RuntimeError(f"Unclassified P12 parameters: {sorted(unknown)}")
    if not phase or not electronics or not head:
        raise RuntimeError(
            "P/E/H partition must contain optical phase, electronic body and head"
        )
    stages = sorted(phase)
    if stages != list(range(len(stages))):
        raise RuntimeError(f"Optical phase stage indices are not contiguous: {stages}")
    result = ParameterPartition(
        phase_by_stage=tuple(tuple(sorted(phase[index])) for index in stages),
        electronics=tuple(sorted(electronics)),
        head=tuple(sorted(head)),
        frozen=tuple(sorted(frozen)),
    )
    named = tuple(
        sorted(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    )
    if tuple(sorted(result.all_parameters)) != named:
        raise RuntimeError("P/E/H partition is not an exact cover of named parameters")
    if len(result.phase_by_stage) != 8:
        raise RuntimeError(
            f"P12 mechanism audit requires eight phase stages, got {len(result.phase_by_stage)}"
        )
    return result


def _clone_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def compose_parameter_state(
    template: nn.Module,
    partition: ParameterPartition,
    donor_states: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    phase_sources: Sequence[str],
    electronics_source: str,
    head_source: str,
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    """Compose parameters while retaining every buffer from ``template``."""

    if len(phase_sources) != len(partition.phase_by_stage):
        raise ValueError("One phase donor is required per optical stage")
    state = _clone_state(template.state_dict())
    parameter_names = set(partition.all_parameters)
    actual_parameter_names = {name for name, _ in template.named_parameters()}
    registered_buffer_names = {name for name, _ in template.named_buffers()}
    buffer_names = tuple(sorted(set(state) - actual_parameter_names))
    nonpersistent_buffer_names = registered_buffer_names - set(buffer_names)
    frozen_names = tuple(sorted(actual_parameter_names - parameter_names))
    if frozen_names != partition.frozen:
        raise RuntimeError("P/E/H composition received a stale frozen-parameter partition")
    if set(state) != actual_parameter_names | set(buffer_names):
        raise RuntimeError("Module state is not exactly parameters plus persistent buffers")
    if nonpersistent_buffer_names & set(state):
        raise RuntimeError("A runtime-only buffer unexpectedly entered the state dict")

    def copy_parameter(name: str, source: str) -> None:
        if source not in donor_states:
            raise KeyError(f"Unknown parameter donor {source!r}")
        donor = donor_states[source]
        if name not in donor:
            raise KeyError(f"Donor {source!r} does not contain {name!r}")
        value = donor[name].detach().cpu()
        if value.shape != state[name].shape or value.dtype != state[name].dtype:
            raise RuntimeError(
                f"Donor tensor mismatch for {name}: "
                f"{tuple(value.shape)}/{value.dtype} != "
                f"{tuple(state[name].shape)}/{state[name].dtype}"
            )
        state[name] = value.clone()

    for stage, names in enumerate(partition.phase_by_stage):
        for name in names:
            copy_parameter(name, str(phase_sources[stage]))
    for name in partition.electronics:
        copy_parameter(name, electronics_source)
    for name in partition.head:
        copy_parameter(name, head_source)

    # These assertions are the central safety property: no donor persistent
    # buffer or frozen Qwen-stem parameter can enter a counterfactual. Runtime-
    # only buffers are absent from every checkpoint and remain on the fresh
    # model because load_state_dict cannot touch them.
    template_state = template.state_dict()
    for name in (*buffer_names, *frozen_names):
        if not torch.equal(state[name], template_state[name].detach().cpu()):
            raise RuntimeError(f"Mechanism composition changed fixed state {name!r}")
    return state, buffer_names


def _tensor_group_sha256(
    state: Mapping[str, torch.Tensor], names: Iterable[str]
) -> str:
    digest = hashlib.sha256()
    for name in sorted(names):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _add_spec(
    specs: dict[tuple[Any, ...], HybridSpec],
    links: list[dict[str, Any]],
    *,
    phase_sources: Sequence[str],
    electronics_source: str,
    head_source: str,
    analysis: str,
    metadata: Mapping[str, Any],
) -> str:
    key = (tuple(phase_sources), electronics_source, head_source)
    if key not in specs:
        specs[key] = HybridSpec(
            state_id=f"state_{len(specs):03d}",
            phase_sources=tuple(phase_sources),
            electronics_source=electronics_source,
            head_source=head_source,
        )
    state_id = specs[key].state_id
    links.append(
        {
            "analysis": analysis,
            "state_id": state_id,
            **dict(metadata),
        }
    )
    return state_id


def build_mechanism_plan(
    methods: Sequence[str] = UPDATE_METHODS,
    *,
    num_stages: int = 8,
) -> tuple[list[HybridSpec], list[dict[str, Any]]]:
    if tuple(methods) != UPDATE_METHODS:
        raise ValueError(f"Mechanism audit locks update methods to {UPDATE_METHODS}")
    if int(num_stages) != 8:
        raise ValueError("P12 depth counterfactual requires exactly eight stages")
    specs: dict[tuple[Any, ...], HybridSpec] = {}
    links: list[dict[str, Any]] = []
    common_phase = ("common",) * num_stages

    # Complete P/E/H factorial for each formal updating endpoint.
    for method in methods:
        for bits in itertools.product((0, 1), repeat=3):
            subset = tuple(
                block for block, enabled in zip(BLOCKS, bits, strict=True) if enabled
            )
            _add_spec(
                specs,
                links,
                phase_sources=(method,) * num_stages if bits[0] else common_phase,
                electronics_source=method if bits[1] else "common",
                head_source=method if bits[2] else "common",
                analysis="factorial",
                metadata={"method": method, "subset": list(subset)},
            )

    # Phase- and electronics-donor matrices. Cache de-duplication means the
    # common reset and diagonal/full states are not evaluated twice.
    donors = ("common", *methods)
    for recipient in methods:
        for donor in donors:
            _add_spec(
                specs,
                links,
                phase_sources=(donor,) * num_stages,
                electronics_source=recipient,
                head_source=recipient,
                analysis="phase_swap",
                metadata={"recipient": recipient, "donor": donor},
            )
            _add_spec(
                specs,
                links,
                phase_sources=(recipient,) * num_stages,
                electronics_source=donor,
                head_source=recipient,
                analysis="electronics_swap",
                metadata={"recipient": recipient, "donor": donor},
            )

    # Separate the connector-affected stages 1--7 from the locally exact last
    # optical stage. These are evaluated for all three update methods.
    for method in methods:
        _add_spec(
            specs,
            links,
            phase_sources=("common",) * (num_stages - 1) + (method,),
            electronics_source=method,
            head_source=method,
            analysis="phase_depth_reset",
            metadata={"method": method, "kept": "stage8_only"},
        )
        _add_spec(
            specs,
            links,
            phase_sources=(method,) * (num_stages - 1) + ("common",),
            electronics_source=method,
            head_source=method,
            analysis="phase_depth_reset",
            metadata={"method": method, "kept": "stages1_to_7_only"},
        )
    ordered = sorted(specs.values(), key=lambda item: item.state_id)
    return ordered, links


def shapley_values(
    values: Mapping[frozenset[str], float],
    blocks: Sequence[str] = BLOCKS,
) -> dict[str, float]:
    players = tuple(blocks)
    expected = {
        frozenset(players[index] for index in range(len(players)) if mask & (1 << index))
        for mask in range(1 << len(players))
    }
    if set(values) != expected:
        missing = sorted(expected - set(values), key=lambda item: (len(item), sorted(item)))
        extra = sorted(set(values) - expected, key=lambda item: (len(item), sorted(item)))
        raise ValueError(f"Shapley value function is incomplete: missing={missing}, extra={extra}")
    count = len(players)
    factorial = math.factorial
    result: dict[str, float] = {}
    for player in players:
        contribution = 0.0
        others = [candidate for candidate in players if candidate != player]
        for size in range(len(others) + 1):
            for subset_values in itertools.combinations(others, size):
                subset = frozenset(subset_values)
                weight = (
                    factorial(size)
                    * factorial(count - size - 1)
                    / factorial(count)
                )
                contribution += weight * (
                    float(values[subset | {player}]) - float(values[subset])
                )
        result[player] = contribution
    return result


def _formal_config_digest(settings: Settings, implementation_digest: str) -> str:
    return sha256_json(
        {
            "settings": settings.to_dict(),
            "implementation_sha256": implementation_digest,
        }
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON mapping at {path}")
    return value


def _assert_checkpoint_state(
    payload: Mapping[str, Any],
    template: nn.Module,
    *,
    name: str,
) -> None:
    state = payload.get("model")
    if not isinstance(state, Mapping):
        raise RuntimeError(f"{name} does not contain a model state")
    expected = template.state_dict()
    if set(state) != set(expected):
        raise RuntimeError(
            f"{name} model keys differ: missing={sorted(set(expected) - set(state))[:8]}, "
            f"unexpected={sorted(set(state) - set(expected))[:8]}"
        )
    for key, expected_value in expected.items():
        value = state[key]
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"{name} state value {key!r} is not a tensor")
        if value.shape != expected_value.shape or value.dtype != expected_value.dtype:
            raise RuntimeError(f"{name} tensor shape/dtype mismatch at {key!r}")
    signature = state.get(_P11_SIGNATURE_KEY)
    if signature is None or not torch.equal(signature.detach().cpu(), _P11_SIGNATURE):
        raise RuntimeError(f"{name} does not retain the strict P11 signature")


def _validate_payload(
    payload: Mapping[str, Any],
    settings: Settings,
    template: nn.Module,
    *,
    manifest_sha256: str,
    common_start_sha256: str | None,
    implementation_digest: str,
    source_checkpoint_sha256: str,
    source_phase_sha256: str,
    frozen_stem_digest: str,
    name: str,
) -> None:
    identity_model = copy.deepcopy(template)
    configure_runtime_feedback(identity_model, settings)
    _validate_checkpoint_identity(
        payload,
        settings,
        manifest_sha256=manifest_sha256,
        config_digest=_formal_config_digest(settings, implementation_digest),
        implementation_digest=implementation_digest,
        source_checkpoint_sha256=source_checkpoint_sha256,
        common_start_sha256=common_start_sha256,
        source_phase_sha256=source_phase_sha256,
        feedback_manifest=identity_model.feedback_manifest(),
        frozen_stem_digest=frozen_stem_digest,
    )
    _assert_checkpoint_state(payload, template, name=name)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _scalar_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if value is None or isinstance(value, (str, int, float, bool))
    }


def _load_formal_artifacts(
    config: Path,
    task: str,
    seed: int,
    endpoint: str,
    *,
    mechanism_data_dir: Path,
) -> tuple[
    Settings,
    Any,
    Any,
    nn.Module,
    ParameterPartition,
    dict[str, Mapping[str, torch.Tensor]],
    dict[str, Any],
]:
    implementation_digest = implementation_sha256()
    noft = load_settings(config, task=task, method="noft", seed=seed)
    noft.validate_runtime_paths()
    complete, reason = completion_reason(noft)
    if not complete:
        raise RuntimeError(f"Formal NoFT is not complete for {task}/seed{seed}: {reason}")

    # Write the independently rebuilt audit manifest only below mechanism root.
    data_settings = replace(
        noft,
        paths=replace(noft.paths, run_dir=mechanism_data_dir.resolve()),
    )
    _, bundle, loaders = build_data(data_settings)
    template = build_model(noft)
    template.eval()
    partition = partition_parameter_names(template)
    source_sha256 = template.source_manifest["sha256"]
    source_report = template.parameter_report()
    source_phase_sha256 = source_report["source_phase_sha256"]
    frozen_stem_digest = module_state_sha256(template.backbone.stem)

    common_path = noft.paths.common_start_checkpoint
    common_sha256 = sha256_file(common_path)
    noft_result_path = noft.output_dir / "result.json"
    noft_result = _load_json(noft_result_path)
    common_payload = torch.load(common_path, map_location="cpu", weights_only=False)
    if (
        common_payload.get("format") != CHECKPOINT_FORMAT
        or common_payload.get("selected_as_common_start") is not True
        or common_payload.get("completed_head_only_epochs")
        != noft.training.head_only_epochs
    ):
        raise RuntimeError("Common checkpoint is not a selected formal NoFT endpoint")
    if int(common_payload.get("epoch", -1)) != int(noft_result.get("best_epoch", -2)):
        raise RuntimeError("Common checkpoint epoch differs from the formal NoFT best")
    if noft_result.get("common_start_sha256") != common_sha256:
        raise RuntimeError("Formal NoFT result does not name the loaded common checkpoint")
    _validate_payload(
        common_payload,
        noft,
        template,
        manifest_sha256=bundle.manifest_sha256,
        common_start_sha256=None,
        implementation_digest=implementation_digest,
        source_checkpoint_sha256=source_sha256,
        source_phase_sha256=source_phase_sha256,
        frozen_stem_digest=frozen_stem_digest,
        name="common_start",
    )

    donor_states: dict[str, Mapping[str, torch.Tensor]] = {
        "common": common_payload["model"]
    }
    identities: dict[str, Any] = {
        "task": task,
        "seed": seed,
        "endpoint": endpoint,
        "dataset_manifest_sha256": bundle.manifest_sha256,
        "source_checkpoint_sha256": source_sha256,
        "source_phase_sha256": source_phase_sha256,
        "common_start_sha256": common_sha256,
        "frozen_stem_state_sha256": frozen_stem_digest,
        "implementation_sha256": implementation_digest,
        "mechanism_script_sha256": sha256_file(Path(__file__)),
        "formal_results": {
            "noft": {
                "result": str(noft_result_path),
                "common_checkpoint": str(common_path),
                "common_checkpoint_sha256": common_sha256,
                "epoch": int(common_payload["epoch"]),
                "config_digest": common_payload.get("config_digest"),
                "feedback": common_payload.get("feedback"),
            }
        },
    }

    # NoFT must leave every reusable P11-body parameter byte-identical to the
    # strict source model. Only the temporary task head is allowed to move.
    template_state = template.state_dict()
    for name in (*partition.phase, *partition.electronics):
        if not torch.equal(common_payload["model"][name], template_state[name].cpu()):
            raise RuntimeError(f"NoFT common changed reusable backbone parameter {name}")

    for method in UPDATE_METHODS:
        settings = load_settings(config, task=task, method=method, seed=seed)
        complete, reason = completion_reason(settings)
        if not complete:
            raise RuntimeError(
                f"Formal {method} is not complete for {task}/seed{seed}: {reason}"
            )
        result = _load_json(settings.output_dir / "result.json")
        checkpoint_path = settings.output_dir / "checkpoints" / f"{endpoint}.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing formal {endpoint} checkpoint: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        _validate_payload(
            payload,
            settings,
            template,
            manifest_sha256=bundle.manifest_sha256,
            common_start_sha256=common_sha256,
            implementation_digest=implementation_digest,
            source_checkpoint_sha256=source_sha256,
            source_phase_sha256=source_phase_sha256,
            frozen_stem_digest=frozen_stem_digest,
            name=f"{method}/{endpoint}",
        )
        epoch = int(payload.get("epoch", -1))
        if endpoint == "last" and epoch != settings.run_epochs:
            raise RuntimeError(f"{method} last checkpoint is epoch {epoch}, not 50")
        if endpoint == "best" and epoch != int(result.get("best_epoch", -1)):
            raise RuntimeError(
                f"{method} best checkpoint epoch {epoch} != result {result.get('best_epoch')}"
            )
        if payload.get("common_start_sha256") != common_sha256:
            raise RuntimeError(f"{method} endpoint did not inherit the strict common")
        donor_states[method] = payload["model"]
        identities["formal_results"][method] = {
            "result": str(settings.output_dir / "result.json"),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "epoch": epoch,
            "config_digest": payload.get("config_digest"),
            "feedback": payload.get("feedback"),
        }

    for label, state in donor_states.items():
        identities.setdefault("parameter_group_sha256", {})[label] = {
            "phase": _tensor_group_sha256(state, partition.phase),
            "electronics": _tensor_group_sha256(state, partition.electronics),
            "head": _tensor_group_sha256(state, partition.head),
        }
    return (
        noft,
        bundle,
        loaders,
        template,
        partition,
        donor_states,
        identities,
    )


def _factorial_shapley_rows(
    links: Sequence[Mapping[str, Any]],
    state_metrics: Mapping[str, Mapping[str, Any]],
    *,
    task: str,
    seed: int,
    endpoint: str,
    primary_metric: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in UPDATE_METHODS:
        method_links = [
            link
            for link in links
            if link["analysis"] == "factorial" and link["method"] == method
        ]
        for metric in (primary_metric, "loss"):
            values: dict[frozenset[str], float] = {}
            for link in method_links:
                subset = frozenset(str(value) for value in link["subset"])
                value = state_metrics[str(link["state_id"])].get(metric)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise RuntimeError(
                        f"Finite factorial metric {metric!r} is missing for {link['state_id']}"
                    )
                values[subset] = float(value)
            shapley = shapley_values(values)
            empty = values[frozenset()]
            full = values[frozenset(BLOCKS)]
            interaction_pe_at_head = (
                values[frozenset(BLOCKS)]
                - values[frozenset(("phase", "head"))]
                - values[frozenset(("electronics", "head"))]
                + values[frozenset(("head",))]
            )
            rows.append(
                {
                    "task": task,
                    "seed": seed,
                    "endpoint": endpoint,
                    "method": method,
                    "metric": metric,
                    "common_value": empty,
                    "full_value": full,
                    "full_minus_common": full - empty,
                    "phase_shapley": shapley["phase"],
                    "electronics_shapley": shapley["electronics"],
                    "head_shapley": shapley["head"],
                    "shapley_residual": (full - empty) - sum(shapley.values()),
                    "phase_electronics_interaction_at_final_head": interaction_pe_at_head,
                }
            )
    return rows


def _finite_metric(
    state_metrics: Mapping[str, Mapping[str, Any]],
    state_id: str,
    metric: str,
) -> float:
    value = state_metrics[state_id].get(metric)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RuntimeError(f"Finite metric {metric!r} is missing for {state_id}")
    return float(value)


def _one_link(
    links: Sequence[Mapping[str, Any]],
    analysis: str,
    **identity: Any,
) -> Mapping[str, Any]:
    matches = [
        link
        for link in links
        if link.get("analysis") == analysis
        and all(link.get(key) == value for key, value in identity.items())
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {analysis} link for {identity}, found {len(matches)}"
        )
    return matches[0]


def _swap_rows(
    links: Sequence[Mapping[str, Any]],
    state_metrics: Mapping[str, Mapping[str, Any]],
    *,
    task: str,
    seed: int,
    endpoint: str,
    primary_metric: str,
) -> list[dict[str, Any]]:
    """Summarize directed donor transfer without assuming linear mechanisms."""

    rows: list[dict[str, Any]] = []
    for analysis, block in (
        ("phase_swap", "phase"),
        ("electronics_swap", "electronics"),
    ):
        for recipient in UPDATE_METHODS:
            common_link = _one_link(
                links, analysis, recipient=recipient, donor="common"
            )
            own_link = _one_link(
                links, analysis, recipient=recipient, donor=recipient
            )
            for metric in (primary_metric, "loss"):
                common_value = _finite_metric(
                    state_metrics, str(common_link["state_id"]), metric
                )
                own_value = _finite_metric(
                    state_metrics, str(own_link["state_id"]), metric
                )
                own_effect = own_value - common_value
                benefit_sign = -1.0 if metric == "loss" else 1.0
                for donor in ("common", *UPDATE_METHODS):
                    link = _one_link(
                        links, analysis, recipient=recipient, donor=donor
                    )
                    value = _finite_metric(
                        state_metrics, str(link["state_id"]), metric
                    )
                    raw_effect = value - common_value
                    rows.append(
                        {
                            "task": task,
                            "seed": seed,
                            "endpoint": endpoint,
                            "analysis": analysis,
                            "exchanged_block": block,
                            "recipient": recipient,
                            "donor": donor,
                            "metric": metric,
                            "state_id": link["state_id"],
                            "value": value,
                            "common_donor_value": common_value,
                            "recipient_own_value": own_value,
                            "raw_delta_over_common_donor": raw_effect,
                            "benefit_delta_over_common_donor": benefit_sign
                            * raw_effect,
                            "raw_delta_from_recipient_own": value - own_value,
                            "benefit_delta_from_recipient_own": benefit_sign
                            * (value - own_value),
                            "transport_ratio_to_recipient_own_effect": (
                                raw_effect / own_effect
                                if abs(own_effect) > 1.0e-12
                                else None
                            ),
                        }
                    )
    return rows


def _phase_depth_rows(
    links: Sequence[Mapping[str, Any]],
    state_metrics: Mapping[str, Mapping[str, Any]],
    *,
    task: str,
    seed: int,
    endpoint: str,
    primary_metric: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in UPDATE_METHODS:
        factorial = [
            link
            for link in links
            if link.get("analysis") == "factorial" and link.get("method") == method
        ]
        phase_reset = next(
            link
            for link in factorial
            if frozenset(link["subset"]) == frozenset(("electronics", "head"))
        )
        full = next(
            link
            for link in factorial
            if frozenset(link["subset"]) == frozenset(BLOCKS)
        )
        early = _one_link(
            links,
            "phase_depth_reset",
            method=method,
            kept="stages1_to_7_only",
        )
        last = _one_link(
            links, "phase_depth_reset", method=method, kept="stage8_only"
        )
        for metric in (primary_metric, "loss"):
            reset_value = _finite_metric(
                state_metrics, str(phase_reset["state_id"]), metric
            )
            full_value = _finite_metric(state_metrics, str(full["state_id"]), metric)
            early_value = _finite_metric(state_metrics, str(early["state_id"]), metric)
            last_value = _finite_metric(state_metrics, str(last["state_id"]), metric)
            full_effect = full_value - reset_value
            rows.append(
                {
                    "task": task,
                    "seed": seed,
                    "endpoint": endpoint,
                    "method": method,
                    "metric": metric,
                    "phase_reset_value": reset_value,
                    "full_phase_value": full_value,
                    "stages1_to_7_only_value": early_value,
                    "stage8_only_value": last_value,
                    "full_phase_raw_effect": full_effect,
                    "stages1_to_7_recovery_ratio": (
                        (early_value - reset_value) / full_effect
                        if abs(full_effect) > 1.0e-12
                        else None
                    ),
                    "stage8_recovery_ratio": (
                        (last_value - reset_value) / full_effect
                        if abs(full_effect) > 1.0e-12
                        else None
                    ),
                    "early_late_interaction": (
                        full_value - early_value - last_value + reset_value
                    ),
                }
            )
    return rows


def run_task_seed_endpoint(
    *,
    config: Path,
    task: str,
    seed: int,
    endpoint: str,
    output_dir: Path,
    split: str,
    max_eval_batches: int | None,
    device: torch.device,
    overwrite: bool,
) -> dict[str, Any]:
    result_path = output_dir / "mechanism_result.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(
            f"Mechanism result already exists: {result_path}; pass --overwrite explicitly"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    (
        settings,
        bundle,
        loaders,
        template,
        partition,
        donor_states,
        identities,
    ) = _load_formal_artifacts(
        config,
        task,
        seed,
        endpoint,
        mechanism_data_dir=output_dir / "rebuilt_data_manifest",
    )
    if split not in loaders:
        raise ValueError(f"Unknown audit split {split!r}")
    specs, links = build_mechanism_plan(num_stages=len(partition.phase_by_stage))
    write_json(
        output_dir / "mechanism_manifest.json",
        {
            "format": MECHANISM_FORMAT,
            "status": "running",
            "identities": identities,
            "split": split,
            "max_eval_batches": max_eval_batches,
            "states": [spec.__dict__ for spec in specs],
            "analysis_links": links,
            "buffers_are_never_exchanged": True,
            "parameter_partition": {
                "phase_by_stage": [list(names) for names in partition.phase_by_stage],
                "electronics": list(partition.electronics),
                "head": list(partition.head),
                "frozen_qwen_stem": list(partition.frozen),
            },
        },
    )

    state_rows: list[dict[str, Any]] = []
    state_metrics: dict[str, Mapping[str, Any]] = {}
    analyses_by_state: dict[str, list[str]] = {}
    for link in links:
        analyses_by_state.setdefault(str(link["state_id"]), []).append(str(link["analysis"]))

    template.to("cpu")
    for index, spec in enumerate(specs, start=1):
        hybrid = copy.deepcopy(template)
        state, buffer_names = compose_parameter_state(
            hybrid,
            partition,
            donor_states,
            phase_sources=spec.phase_sources,
            electronics_source=spec.electronics_source,
            head_source=spec.head_source,
        )
        hybrid.load_state_dict(state, strict=True)
        retained_buffer_count = len(buffer_names)
        del state
        if module_state_sha256(hybrid.backbone.stem) != identities[
            "frozen_stem_state_sha256"
        ]:
            raise RuntimeError(f"Hybrid {spec.state_id} changed the frozen Qwen stem")
        hybrid.to(device)
        metrics = evaluate(
            hybrid,
            loaders[split],
            settings,
            device,
            ablation="normal",
            max_batches=max_eval_batches,
            include_retrieval=False,
        )
        primary = metrics.get(settings.task_settings.primary_metric)
        if not isinstance(primary, (int, float)) or not math.isfinite(float(primary)):
            raise RuntimeError(
                f"Hybrid {spec.state_id} produced no finite primary metric"
            )
        state_metrics[spec.state_id] = metrics
        row: dict[str, Any] = {
            "task": task,
            "seed": seed,
            "endpoint": endpoint,
            "split": split,
            "state_id": spec.state_id,
            "phase_sources": ",".join(spec.phase_sources),
            "electronics_source": spec.electronics_source,
            "head_source": spec.head_source,
            "analyses": ",".join(sorted(set(analyses_by_state[spec.state_id]))),
            "primary_metric": settings.task_settings.primary_metric,
            "primary_value": float(primary),
            "buffer_count_retained_from_fresh_model": retained_buffer_count,
            **_scalar_metrics(metrics),
        }
        state_rows.append(row)
        write_json(output_dir / "states_partial.json", state_rows)
        print(
            f"[P12 mechanism] {task}/seed{seed}/{endpoint} "
            f"state={index}/{len(specs)} {spec.state_id} "
            f"{settings.task_settings.primary_metric}={float(primary):.6f}",
            flush=True,
        )
        del hybrid
        if device.type == "cuda":
            torch.cuda.empty_cache()

    shapley_rows = _factorial_shapley_rows(
        links,
        state_metrics,
        task=task,
        seed=seed,
        endpoint=endpoint,
        primary_metric=settings.task_settings.primary_metric,
    )
    swap_rows = _swap_rows(
        links,
        state_metrics,
        task=task,
        seed=seed,
        endpoint=endpoint,
        primary_metric=settings.task_settings.primary_metric,
    )
    phase_depth_rows = _phase_depth_rows(
        links,
        state_metrics,
        task=task,
        seed=seed,
        endpoint=endpoint,
        primary_metric=settings.task_settings.primary_metric,
    )
    result = {
        "format": MECHANISM_FORMAT,
        "status": "complete",
        "task": task,
        "seed": seed,
        "endpoint": endpoint,
        "split": split,
        "max_eval_batches": max_eval_batches,
        "primary_metric": settings.task_settings.primary_metric,
        "identities": identities,
        "dataset_counts": bundle.metadata.get("counts"),
        "buffers_are_never_exchanged": True,
        "states": state_rows,
        "state_metrics": state_metrics,
        "analysis_links": links,
        "shapley": shapley_rows,
        "directed_swaps": swap_rows,
        "phase_depth": phase_depth_rows,
        "interpretation_boundary": (
            "Fixed feedback changes optical inter-stage error connectors only. "
            "Local current-phase gradients and all electronic/head gradients remain exact."
        ),
    }
    write_json(result_path, result)
    _write_csv(output_dir / "states.csv", state_rows)
    _write_csv(output_dir / "shapley.csv", shapley_rows)
    _write_csv(output_dir / "directed_swaps.csv", swap_rows)
    _write_csv(output_dir / "phase_depth.csv", phase_depth_rows)
    write_json(
        output_dir / "mechanism_manifest.json",
        {
            "format": MECHANISM_FORMAT,
            "status": "complete",
            "result": str(result_path),
            "identities": identities,
            "state_count": len(state_rows),
            "buffers_are_never_exchanged": True,
        },
    )
    return result


def _parse_csv_choices(value: str, allowed: Sequence[str], *, name: str) -> tuple[str, ...]:
    result = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = set(result) - set(allowed)
    if not result or unknown or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError(
            f"{name} must be a unique comma list drawn from {tuple(allowed)}; "
            f"unknown={sorted(unknown)}"
        )
    return result


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error
    if not seeds or any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be unique non-negative integers")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Zero-training P12 P/E/H factorial, swap and phase-depth mechanism audit"
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--formal-repo-root",
        type=Path,
        help=(
            "absolute root of the locked training worktree; use this when the "
            "audit code lives in a derived worktree so settings/config digests "
            "are reconstructed with the original absolute paths"
        ),
    )
    parser.add_argument(
        "--tasks",
        default="caltech101,isic2016",
        help="comma list; pilot default covers classification and dense prediction",
    )
    parser.add_argument("--seeds", type=_parse_seeds, default=(2026,))
    parser.add_argument(
        "--endpoints",
        default="best",
        help="best, last, or best,last; use both to audit validation selection",
    )
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    formal_repo_root: Path | None = None
    if args.formal_repo_root is not None:
        formal_repo_root = args.formal_repo_root.expanduser().resolve()
        try:
            formal_implementation_files = (
                settings_module.implementation_files_for_repository(
                    formal_repo_root
                )
            )
        except FileNotFoundError as error:
            raise FileNotFoundError(
                "Formal repo root does not contain the locked P12 implementation; "
                f"{error}"
            ) from error
        # ``load_settings`` and ``implementation_sha256`` are imported function
        # objects whose globals remain in ``settings_module``. Updating this one
        # module-level root and its matching physical-path ledger therefore
        # reconstruct both the absolute Settings paths and implementation hash
        # exactly as they were during training, while mechanism.py itself can
        # remain in an isolated derived worktree.
        settings_module.REPO_ROOT = formal_repo_root
        settings_module.IMPLEMENTATION_FILES = formal_implementation_files
    tasks = _parse_csv_choices(args.tasks, TASKS, name="tasks")
    endpoints = _parse_csv_choices(args.endpoints, ("best", "last"), name="endpoints")
    if args.max_eval_batches is not None and args.max_eval_batches <= 0:
        raise ValueError("--max-eval-batches must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA mechanism audit but CUDA is unavailable")
    device = torch.device(args.device)
    config = args.config.expanduser().resolve()
    probe = load_settings(config, task=tasks[0], method="noft", seed=args.seeds[0])
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else probe.paths.output_root.parent
        / f"{probe.paths.output_root.name}_mechanism"
    )
    summary_path = output_root / "summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Mechanism summary already exists: {summary_path}; "
            "use a new output root or pass --overwrite explicitly"
        )
    results: list[dict[str, Any]] = []
    for task in tasks:
        for seed in args.seeds:
            for endpoint in endpoints:
                destination = output_root / task / f"seed_{seed}" / endpoint
                results.append(
                    run_task_seed_endpoint(
                        config=config,
                        task=task,
                        seed=seed,
                        endpoint=endpoint,
                        output_dir=destination,
                        split=args.split,
                        max_eval_batches=args.max_eval_batches,
                        device=device,
                        overwrite=bool(args.overwrite),
                    )
                )
    summary = {
        "format": MECHANISM_FORMAT,
        "status": "complete",
        "config": str(config),
        "formal_repo_root": str(formal_repo_root) if formal_repo_root else None,
        "tasks": list(tasks),
        "seeds": list(args.seeds),
        "endpoints": list(endpoints),
        "split": args.split,
        "output_root": str(output_root),
        "runs": [
            {
                "task": result["task"],
                "seed": result["seed"],
                "endpoint": result["endpoint"],
                "state_count": len(result["states"]),
            }
            for result in results
        ],
    }
    write_json(summary_path, summary)
    print(f"[P12 mechanism] complete: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BLOCKS",
    "HybridSpec",
    "MECHANISM_FORMAT",
    "ParameterPartition",
    "build_mechanism_plan",
    "build_parser",
    "compose_parameter_state",
    "main",
    "partition_parameter_names",
    "run_task_seed_endpoint",
    "shapley_values",
]
