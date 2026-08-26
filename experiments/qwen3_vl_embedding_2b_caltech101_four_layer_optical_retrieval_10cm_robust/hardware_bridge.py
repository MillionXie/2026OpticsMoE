from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageOps

from experiments.hardware_sdk.workflows.reconstruct_slm import (
    encode_active_amplitude_with_metadata,
    encode_active_phase,
    reconstruct_directory,
    save_active_png,
)
from experiments.qwen3_vl_embedding_2b_caltech101_language2_optical_retrieval.hardware_bridge import (
    load_ccd,
)
from experiments.qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval.prepare_caltech101_retrieval import (
    prepare_caltech101_subset,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.features import (
    move_inputs,
    preprocess_images,
    student_embeddings,
    validate_token_budgets,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.io_utils import (
    seed_everything,
    write_csv,
    write_json,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optical_artifacts import (
    phase_tensors,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.retrieval_metrics import (
    evaluate_embeddings,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.train_optical_retrieval import (
    episodic_prototype_retrieval_loss,
    load_checkpoint,
    supervised_contrastive_loss,
)

from .modeling import build_hybrid_student, load_backbone
from .settings import load_settings


STAGES = (
    "vision_expert",
    "vision_global",
    "language_expert",
    "language_global",
)
UPSTREAM_SOURCES = ("measured", "simulation")


def _replacement_modules(replacement: Any) -> list[torch.nn.Module]:
    modules = [replacement.vision_surrogate, replacement.language_surrogate]
    modules.extend(
        module
        for module in (
            getattr(replacement, "vision_pre_attention", None),
            getattr(replacement, "language_pre_attention", None),
        )
        if module is not None
    )
    return modules


def _replacement_parameters(replacement: Any) -> list[torch.nn.Parameter]:
    seen: set[int] = set()
    values: list[torch.nn.Parameter] = []
    for module in _replacement_modules(replacement):
        for parameter in module.parameters():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                values.append(parameter)
    return values


def _set_replacement_eval(replacement: Any) -> None:
    for module in _replacement_modules(replacement):
        module.eval()


def _key(sample: Any) -> str:
    digest = hashlib.sha1(sample.sample_id.encode("utf-8")).hexdigest()[:10]
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in sample.sku_name)
    return f"{sample.split}__{sample.sku_index:02d}__{safe}__{digest}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage_dir(session_dir: Path, stage: str) -> Path:
    return session_dir / f"{STAGES.index(stage) + 1:02d}_{stage}"


def _samples(bundle: Any) -> list[Any]:
    return list(bundle.all_samples())


def _read_manifest(session_dir: Path) -> list[dict[str, str]]:
    with (session_dir / "manifest.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        return list(csv.DictReader(handle))


def _load_model(settings: Any, checkpoint: Path):
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        configured_output = Path(settings.output_dir).expanduser().resolve()
        runs_root = configured_output.parent
        candidates = sorted(runs_root.glob("*/ema_best_train_loss_checkpoint.pt"))
        candidate_text = "\n".join(f"  - {path}" for path in candidates[:12])
        if not candidate_text:
            candidate_text = "  (no EMA checkpoint found under the experiment runs directory)"
        raise FileNotFoundError(
            f"Four-layer checkpoint is missing: {checkpoint}\n"
            f"This YAML resolves training.output_dir to: {configured_output}\n"
            "Available sibling EMA checkpoints are listed only for diagnosis; "
            "do not mix a checkpoint with a different YAML:\n"
            f"{candidate_text}"
        )
    device = torch.device(
        settings.device
        if settings.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    loaded = load_backbone(settings, device)
    settings.resolve_architecture(loaded.model)
    replacement, readout = build_hybrid_student(loaded, settings)
    load_checkpoint(checkpoint, replacement, readout)
    replacement.set_phase_dropout_active(False)
    _set_replacement_eval(replacement)
    readout.eval()
    return loaded, replacement, readout


def _load_stage_ccd(
    settings: Any, session_dir: Path, stage: str, key: str
) -> torch.Tensor:
    return load_ccd(
        _stage_dir(session_dir, stage),
        key,
        use_simulation=False,
        settings=settings,
        persist_registered=False,
        reuse_registered=False,
    )


def _measurement_plan(
    stage: str,
    upstream_source: str,
    *,
    include_current: bool,
) -> tuple[str, ...]:
    if upstream_source not in UPSTREAM_SOURCES:
        raise ValueError(
            f"upstream_source must be one of {UPSTREAM_SOURCES}; "
            f"got {upstream_source!r}"
        )
    stage_index = STAGES.index(stage)
    if upstream_source == "measured":
        stop = stage_index + (1 if include_current else 0)
        return tuple(STAGES[:stop])
    return (stage,) if include_current else ()


def _install_measurements(
    replacement: Any,
    settings: Any,
    session_dir: Path,
    keys: list[str],
    *,
    measured_stages: tuple[str, ...],
) -> None:
    unknown = set(measured_stages).difference(STAGES)
    if unknown:
        raise ValueError(f"Unknown measured optical stages: {sorted(unknown)}")
    selected = set(measured_stages)
    tensors: dict[str, torch.Tensor | None] = {}
    device = next(replacement.vision_surrogate.parameters()).device
    for stage in STAGES:
        tensors[stage] = (
            torch.stack(
                [_load_stage_ccd(settings, session_dir, stage, key) for key in keys]
            ).to(device)
            if stage in selected
            else None
        )
    replacement.vision_surrogate.core.optical_branch.set_measured_ccd(
        expert=tensors["vision_expert"], global_=tensors["vision_global"]
    )
    replacement.language_surrogate.core.optical_branch.set_measured_ccd(
        expert=tensors["language_expert"], global_=tensors["language_global"]
    )


def _clear_measurements(replacement: Any) -> None:
    replacement.vision_surrogate.core.optical_branch.clear_measured_ccd()
    replacement.language_surrogate.core.optical_branch.clear_measured_ccd()


def _images(samples: Iterable[Any], size: int) -> list[Image.Image]:
    result: list[Image.Image] = []
    for sample in samples:
        with Image.open(sample.image_path) as source:
            result.append(
                ImageOps.fit(
                    source.convert("RGB"),
                    (size, size),
                    method=Image.Resampling.BICUBIC,
                )
            )
    return result


def _forward_samples(
    loaded: Any,
    replacement: Any,
    readout: Any,
    settings: Any,
    samples: list[Any],
) -> torch.Tensor:
    inputs = preprocess_images(
        loaded.processor, _images(samples, settings.image_size), settings.instruction
    )
    validate_token_budgets(inputs, settings)
    return student_embeddings(
        loaded.model, replacement, readout, move_inputs(inputs, loaded.device)
    )[0]


def _branch_for_stage(replacement: Any, stage: str):
    core = (
        replacement.vision_surrogate.core
        if stage.startswith("vision")
        else replacement.language_surrogate.core
    )
    return core.optical_branch


def _amplitude_for_stage(branch: Any, stage: str) -> torch.Tensor:
    value = (
        branch.last_expert_input_amplitude
        if stage.endswith("expert")
        else branch.last_global_input_amplitude
    )
    if value is None:
        raise RuntimeError(f"No optical input was captured for {stage}")
    return value


def _phase_for_stage(replacement: Any, stage: str, settings: Any) -> np.ndarray:
    branch = _branch_for_stage(replacement, stage)
    phases = phase_tensors(branch.core)
    value = (
        phases["physical_expert_mosaic_rad"]
        if stage.endswith("expert")
        else phases["physical_global_phase_rad"]
    )
    if settings.hardware_phase_flip_vertical:
        value = torch.flip(value, (-2,))
    if settings.hardware_phase_flip_horizontal:
        value = torch.flip(value, (-1,))
    return encode_active_phase(value.detach().cpu().numpy())


@torch.no_grad()
def export_stage(
    settings: Any,
    checkpoint: Path,
    session_dir: Path,
    stage: str,
    upstream_source: str = "measured",
) -> None:
    stage_index = STAGES.index(stage)
    measured_upstream_stages = _measurement_plan(
        stage, upstream_source, include_current=False
    )
    bundle = prepare_caltech101_subset(settings, persist=True)
    samples = _samples(bundle)
    loaded, replacement, readout = _load_model(settings, checkpoint)
    session_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "order": index,
            "key": _key(sample),
            "sample_id": sample.sample_id,
            "split": sample.split,
            "sku_index": sample.sku_index,
            "sku_name": sample.sku_name,
            "image_path": str(sample.image_path),
        }
        for index, sample in enumerate(samples)
    ]
    if not (session_dir / "manifest.csv").is_file():
        write_csv(session_dir / "manifest.csv", rows, list(rows[0]))
    elif [row["key"] for row in _read_manifest(session_dir)] != [
        row["key"] for row in rows
    ]:
        raise RuntimeError("Existing hardware manifest does not match this dataset")
    destination = _stage_dir(session_dir, stage)
    compact = destination / "compact_amplitude"
    compact.mkdir(parents=True, exist_ok=True)
    (destination / "ccd_captured").mkdir(exist_ok=True)
    compact_phase_path = destination / "compact_phase" / f"{stage}.png"
    save_active_png(
        _phase_for_stage(replacement, stage, settings),
        compact_phase_path,
    )
    reconstruct_directory(
        compact_phase_path.parent,
        destination / "phase_to_play",
        slm_size_wh=(
            settings.hardware_phase_slm_width,
            settings.hardware_phase_slm_height,
        ),
        scale_factor=None,
        center_xy=(
            settings.hardware_phase_slm_center_x,
            settings.hardware_phase_slm_center_y,
        ),
        logical_pixel_pitch_um=settings.language_optical_pixel_pitch_um,
        slm_pixel_pitch_um=settings.hardware_phase_slm_pixel_pitch_um,
    )
    amplitude_rows: list[dict[str, Any]] = []
    try:
        for start in range(0, len(samples), 10):
            batch = samples[start : start + 10]
            keys = [_key(sample) for sample in batch]
            _install_measurements(
                replacement,
                settings,
                session_dir,
                keys,
                measured_stages=measured_upstream_stages,
            )
            _forward_samples(loaded, replacement, readout, settings, batch)
            amplitude = _amplitude_for_stage(
                _branch_for_stage(replacement, stage), stage
            ).detach().cpu()
            for key, value in zip(keys, amplitude):
                encoded, encoding = encode_active_amplitude_with_metadata(
                    value.numpy()
                )
                path = compact / f"{key}.png"
                save_active_png(
                    encoded, path
                )
                amplitude_rows.append(
                    {
                        "key": key,
                        "filename": path.name,
                        "sha256": _sha256(path),
                        **encoding,
                    }
                )
            print(
                f"[export_{stage}] {min(start + len(batch), len(samples))}/{len(samples)}",
                flush=True,
            )
    finally:
        _clear_measurements(replacement)
        replacement.close()
    write_csv(
        destination / "compact_amplitude_manifest.csv",
        amplitude_rows,
        list(amplitude_rows[0]),
    )
    write_json(
        destination / "transport_spec.json",
        {
            "schema_version": 2,
            "stage": stage,
            "upstream_source": upstream_source,
            "measured_upstream_stages": list(measured_upstream_stages),
            "simulated_upstream_stages": [
                name for name in STAGES[:stage_index]
                if name not in measured_upstream_stages
            ],
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "samples": len(samples),
            "compact_amplitude": "478x478 uint8 PNG in model coordinates",
            "compact_phase": "478x478 uint8 PNG in configured export orientation",
            "laboratory_reconstruction": {
                "amplitude": {
                    "logical_pixel_pitch_um": settings.language_optical_pixel_pitch_um,
                    "slm_pixel_pitch_um": settings.hardware_amplitude_slm_pixel_pitch_um,
                    "slm_size_wh": [
                        settings.hardware_amplitude_slm_width,
                        settings.hardware_amplitude_slm_height,
                    ],
                    "center_xy": [
                        settings.hardware_amplitude_slm_center_x,
                        settings.hardware_amplitude_slm_center_y,
                    ],
                    "bright_value_uint8": settings.hardware_amplitude_bright_value_uint8,
                    "dark_value_uint8": settings.hardware_amplitude_dark_value_uint8,
                    "invert_before_export": settings.hardware_amplitude_invert_before_export,
                    "rule": (
                        "one logical pixel to one native pixel"
                        if settings.hardware_amplitude_slm_pixel_pitch_um
                        == settings.language_optical_pixel_pitch_um
                        else "centered physical-coordinate nearest raster"
                    ),
                },
                "phase": {
                    "logical_pixel_pitch_um": settings.language_optical_pixel_pitch_um,
                    "slm_pixel_pitch_um": settings.hardware_phase_slm_pixel_pitch_um,
                    "slm_size_wh": [
                        settings.hardware_phase_slm_width,
                        settings.hardware_phase_slm_height,
                    ],
                    "center_xy": [
                        settings.hardware_phase_slm_center_x,
                        settings.hardware_phase_slm_center_y,
                    ],
                    "rule": "centered physical-coordinate nearest raster",
                    "flip_vertical_before_raster": settings.hardware_phase_flip_vertical,
                    "flip_horizontal_before_raster": settings.hardware_phase_flip_horizontal,
                },
            },
            "expected_ccd_upload": "478x478 uint8 grayscale PNG; no flip",
            "server_persistence": "no simulation CCD and no per-sample float32 PT cache",
        },
    )


def _enable(modules: list[torch.nn.Module]) -> list[torch.nn.Parameter]:
    seen: set[int] = set()
    result: list[torch.nn.Parameter] = []
    for module in modules:
        module.requires_grad_(True)
        for parameter in module.parameters():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                result.append(parameter)
    return result


def _capture_contract_parameters(
    replacement: Any, stage: str
) -> dict[str, torch.nn.Parameter]:
    """Parameters that must stay frozen after the stage payload was captured.

    The contract covers every module that determined either an already-played
    amplitude frame or an already-captured CCD frame. Updating one of these
    tensors without replaying the SLM would make the injected measurement no
    longer correspond to the model's current upstream computation.
    """

    if stage not in STAGES:
        raise ValueError(f"Unknown optical stage {stage!r}")
    v = replacement.vision_surrogate.core
    l = replacement.language_surrogate.core
    vb = v.optical_branch
    lb = l.optical_branch
    named: dict[str, torch.nn.Parameter] = {}
    seen: set[int] = set()

    def add_module(label: str, module: torch.nn.Module | None) -> None:
        if module is None:
            return
        for name, parameter in module.named_parameters():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            suffix = f".{name}" if name else ""
            named[f"{label}{suffix}"] = parameter

    def add_parameter(label: str, parameter: torch.nn.Parameter) -> None:
        if id(parameter) in seen:
            return
        seen.add(id(parameter))
        named[label] = parameter

    def add_vision_expert_contract() -> None:
        add_module("vision.input_adapter", v.input_adapter)
        add_module("vision.input_norm", v.input_norm)
        add_module("vision.optical.input_adapter", vb.core.input_adapter)
        add_module("vision.optical.input_norm", vb.core.input_norm)
        add_module("vision.router", vb.core.router)
        add_module("vision.expert_phase", vb.core.expert_layers)
        add_module(
            "vision.pre_attention",
            getattr(replacement, "vision_pre_attention", None),
        )

    def add_vision_global_contract() -> None:
        add_vision_expert_contract()
        add_module("vision.block1", v.blocks[0])
        add_module("vision.expert_readout", vb.expert_readout)
        add_module("vision.expert_output_adapter", vb.expert_output_adapter)
        add_parameter("vision.block1_optical_fusion_logit", v.block1_optical_fusion_logit)
        add_module("vision.global_phase", vb.core.global_phase)

    def add_language_expert_contract() -> None:
        # Language payloads consume the complete frozen Vision result.
        add_module("vision.surrogate", replacement.vision_surrogate)
        add_module(
            "vision.pre_attention",
            getattr(replacement, "vision_pre_attention", None),
        )
        add_module("language.input_adapter", l.input_adapter)
        add_module("language.input_norm", l.input_norm)
        add_module("language.optical.input_adapter", lb.core.input_adapter)
        add_module("language.optical.input_norm", lb.core.input_norm)
        add_module("language.router", lb.core.router)
        add_module("language.expert_phase", lb.core.expert_layers)
        add_module(
            "language.pre_attention",
            getattr(replacement, "language_pre_attention", None),
        )

    def add_language_global_contract() -> None:
        add_language_expert_contract()
        add_module("language.block1", l.blocks[0])
        add_module("language.expert_readout", lb.expert_readout)
        add_module("language.expert_output_adapter", lb.expert_output_adapter)
        add_parameter(
            "language.block1_optical_fusion_logit",
            l.block1_optical_fusion_logit,
        )
        add_module("language.global_phase", lb.core.global_phase)

    if stage == "vision_expert":
        add_vision_expert_contract()
    elif stage == "vision_global":
        add_vision_global_contract()
    elif stage == "language_expert":
        add_language_expert_contract()
    else:
        add_language_global_contract()
    return named


def _assert_capture_contract_frozen(replacement: Any, stage: str) -> None:
    violations = [
        name
        for name, parameter in _capture_contract_parameters(replacement, stage).items()
        if parameter.requires_grad
    ]
    if violations:
        preview = ", ".join(violations[:12])
        suffix = " ..." if len(violations) > 12 else ""
        raise RuntimeError(
            f"Hardware capture contract for {stage} would be invalidated by "
            f"trainable upstream tensors: {preview}{suffix}"
        )


def _downstream_parameters(replacement: Any, readout: Any, stage: str):
    if stage not in STAGES:
        raise ValueError(f"Unknown optical stage {stage!r}")
    for module in _replacement_modules(replacement):
        module.requires_grad_(False)
    readout.requires_grad_(False)
    v = replacement.vision_surrogate.core
    l = replacement.language_surrogate.core
    vb = v.optical_branch
    lb = l.optical_branch
    modules: list[torch.nn.Module] = []
    if stage == "vision_expert":
        modules += [
            v.blocks,
            vb.expert_readout,
            vb.expert_output_adapter,
            vb.core.global_phase,
            vb.core.readout,
            vb.core.output_adapter,
            v.output_norm,
            v.output_adapter,
            l,
        ]
        v.block1_optical_fusion_logit.requires_grad_(True)
        v.block2_optical_fusion_logit.requires_grad_(True)
    elif stage == "vision_global":
        modules += [
            v.blocks[1],
            vb.core.readout,
            vb.core.output_adapter,
            v.output_norm,
            v.output_adapter,
            l,
        ]
        v.block2_optical_fusion_logit.requires_grad_(True)
    elif stage == "language_expert":
        modules += [
            l.blocks,
            lb.expert_readout,
            lb.expert_output_adapter,
            lb.core.global_phase,
            lb.core.readout,
            lb.core.output_adapter,
            l.output_norm,
        ]
        l.block1_optical_fusion_logit.requires_grad_(True)
        l.block2_optical_fusion_logit.requires_grad_(True)
    else:
        modules += [
            l.blocks[1],
            lb.core.readout,
            lb.core.output_adapter,
            l.output_norm,
        ]
        l.block2_optical_fusion_logit.requires_grad_(True)
    modules.append(readout)
    parameters = _enable(modules)
    # Retrieval reads Language detector features before this final hidden
    # adapter, so it is intentionally frozen in both joint and hardware runs.
    l.output_adapter.requires_grad_(False)
    l.residual_logit.requires_grad_(False)
    parameters = [parameter for parameter in parameters if parameter.requires_grad]
    # Scalar gates are not modules.
    for parameter in _replacement_parameters(replacement):
        if parameter.requires_grad and all(id(parameter) != id(item) for item in parameters):
            parameters.append(parameter)
    _assert_capture_contract_frozen(replacement, stage)
    return parameters


def _downstream_optimizer(
    parameters: list[torch.nn.Parameter],
    replacement: Any,
    readout: Any,
    settings: Any,
) -> torch.optim.Optimizer:
    trainable_ids = {id(parameter) for parameter in parameters}
    phase_ids = {
        id(parameter)
        for group in replacement.phase_parameter_groups().values()
        for parameter in group
        if id(parameter) in trainable_ids
    }
    router_ids = {
        id(parameter)
        for parameter in replacement.router_parameters()
        if id(parameter) in trainable_ids
    }
    readout_ids = {
        id(parameter)
        for parameter in readout.parameters()
        if id(parameter) in trainable_ids
    }
    if (phase_ids & router_ids) or (phase_ids & readout_ids) or (router_ids & readout_ids):
        raise RuntimeError("Hardware fine-tune optimizer groups overlap")
    reserved = phase_ids | router_ids | readout_ids
    grouped = [
        (
            "downstream_electronic",
            [parameter for parameter in parameters if id(parameter) not in reserved],
            float(settings.learning_rate),
        ),
        (
            "downstream_phases",
            [parameter for parameter in parameters if id(parameter) in phase_ids],
            float(settings.phase_learning_rate),
        ),
        (
            "downstream_routers",
            [parameter for parameter in parameters if id(parameter) in router_ids],
            float(settings.router_learning_rate),
        ),
        (
            "retrieval_readout",
            [parameter for parameter in parameters if id(parameter) in readout_ids],
            float(settings.readout_learning_rate),
        ),
    ]
    optimizer_groups = [
        {"params": values, "lr": learning_rate, "group_name": name}
        for name, values, learning_rate in grouped
        if values
    ]
    covered = {
        id(parameter)
        for group in optimizer_groups
        for parameter in group["params"]
    }
    if covered != trainable_ids:
        raise RuntimeError("Hardware fine-tune optimizer does not cover every tensor")
    return torch.optim.AdamW(optimizer_groups, weight_decay=settings.weight_decay)


def _batch_embeddings(
    loaded: Any,
    replacement: Any,
    readout: Any,
    settings: Any,
    session_dir: Path,
    samples: list[Any],
    measured_stages: tuple[str, ...],
) -> torch.Tensor:
    _install_measurements(
        replacement,
        settings,
        session_dir,
        [_key(sample) for sample in samples],
        measured_stages=measured_stages,
    )
    return _forward_samples(loaded, replacement, readout, settings, samples)


def _set_hardware_finetune_mode(replacement: Any, readout: Any, stage: str) -> None:
    """Keep measured stages deterministic and perturb only simulated optics downstream."""
    _set_replacement_eval(replacement)
    replacement.set_phase_dropout_active(False)
    readout.train()
    downstream_branches = []
    if stage == "vision_expert":
        downstream_branches = [
            replacement.vision_surrogate.core.optical_branch,
            replacement.language_surrogate.core.optical_branch,
        ]
    elif stage == "vision_global":
        downstream_branches = [replacement.language_surrogate.core.optical_branch]
    elif stage == "language_expert":
        downstream_branches = [replacement.language_surrogate.core.optical_branch]
    for branch in downstream_branches:
        branch.train()
        branch.set_phase_dropout_active(True)


@torch.no_grad()
def _hardware_embeddings(
    loaded: Any,
    replacement: Any,
    readout: Any,
    settings: Any,
    session_dir: Path,
    samples: list[Any],
    measured_stages: tuple[str, ...],
) -> torch.Tensor:
    chunks = []
    _set_replacement_eval(replacement)
    replacement.set_phase_dropout_active(False)
    readout.eval()
    for start in range(0, len(samples), 10):
        chunks.append(
            _batch_embeddings(
                loaded,
                replacement,
                readout,
                settings,
                session_dir,
                samples[start : start + 10],
                measured_stages,
            ).detach().cpu()
        )
    return torch.cat(chunks, dim=0)


def finetune_stage(
    settings: Any,
    checkpoint: Path,
    session_dir: Path,
    stage: str,
    epochs: int,
    upstream_source: str = "measured",
) -> None:
    measured_stages = _measurement_plan(
        stage, upstream_source, include_current=True
    )
    bundle = prepare_caltech101_subset(settings, persist=True)
    samples = _samples(bundle)
    loaded, replacement, readout = _load_model(settings, checkpoint)
    parameters = _downstream_parameters(replacement, readout, stage)
    optimizer = _downstream_optimizer(parameters, replacement, readout, settings)
    grouped: dict[int, list[Any]] = defaultdict(list)
    for sample in samples:
        if sample.split == "train":
            grouped[int(sample.sku_index)].append(sample)
    if len(grouped) != 10 or any(len(group) < 3 for group in grouped.values()):
        raise RuntimeError("Four-layer fine-tuning requires 10 classes and 3 train captures")
    generator = torch.Generator().manual_seed(settings.random_seed)
    best = float("inf")
    output = session_dir / "checkpoints" / f"after_{stage}.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for epoch in range(1, epochs + 1):
            _set_hardware_finetune_mode(replacement, readout, stage)
            steps = max(1, sum(map(len, grouped.values())) // 30)
            total = 0.0
            total_contrastive = 0.0
            total_prototype = 0.0
            total_router_balance = 0.0
            total_router_importance = 0.0
            for _ in range(steps):
                batch: list[Any] = []
                for label in sorted(grouped):
                    indexes = torch.randperm(len(grouped[label]), generator=generator)[:3]
                    batch.extend(grouped[label][int(index)] for index in indexes)
                embeddings = _batch_embeddings(
                    loaded,
                    replacement,
                    readout,
                    settings,
                    session_dir,
                    batch,
                    measured_stages,
                )
                labels = torch.tensor(
                    [sample.sku_index for sample in batch], device=embeddings.device
                )
                contrastive = supervised_contrastive_loss(
                    embeddings, labels, settings.temperature
                )
                prototype, _, _ = episodic_prototype_retrieval_loss(
                    embeddings, labels, settings.gallery_temperature
                )
                router_losses = replacement.router_losses()
                router_balance = 0.5 * (
                    router_losses["vision_balance"]
                    + router_losses["language_balance"]
                )
                router_importance = 0.5 * (
                    router_losses["vision_importance"]
                    + router_losses["language_importance"]
                )
                loss = (
                    contrastive
                    + prototype
                    + settings.lambda_router_balance * router_balance
                    + settings.lambda_router_importance * router_importance
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    parameters, 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                total += float(loss.detach())
                total_contrastive += float(contrastive.detach())
                total_prototype += float(prototype.detach())
                total_router_balance += float(router_balance.detach())
                total_router_importance += float(router_importance.detach())
            average = total / steps
            average_contrastive = total_contrastive / steps
            average_prototype = total_prototype / steps
            average_router_balance = total_router_balance / steps
            average_router_importance = total_router_importance / steps
            print(
                f"[finetune_{stage}] epoch={epoch:03d}/{epochs:03d} "
                f"loss={average:.5f} supcon={average_contrastive:.5f} "
                f"prototype={average_prototype:.5f} "
                f"router_balance={average_router_balance:.5f} "
                f"router_importance={average_router_importance:.5f}",
                flush=True,
            )
            if average < best:
                best = average
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                payload["vision_optical"] = {
                    name: value.detach().cpu()
                    for name, value in replacement.vision_surrogate.state_dict().items()
                }
                payload["language_optical"] = {
                    name: value.detach().cpu()
                    for name, value in replacement.language_surrogate.state_dict().items()
                }
                payload["retrieval_readout"] = {
                    name: value.detach().cpu() for name, value in readout.state_dict().items()
                }
                payload["hardware_finetune"] = {
                    "stage": stage,
                    "source_checkpoint": str(checkpoint),
                    "epoch": epoch,
                    "train_loss": average,
                    "train_loss_components": {
                        "supervised_contrastive": average_contrastive,
                        "episodic_prototype": average_prototype,
                        "router_balance": average_router_balance,
                        "router_importance": average_router_importance,
                        "lambda_router_balance": settings.lambda_router_balance,
                        "lambda_router_importance": settings.lambda_router_importance,
                    },
                    "measured_stages": list(measured_stages),
                    "upstream_source": upstream_source,
                    "rule": (
                        "only the selected measured stage and its downstream "
                        "electronic and uncaptured optical modules are adapted; "
                        "all earlier optical stages remain simulated"
                        if upstream_source == "simulation"
                        else (
                            "only downstream electronic and uncaptured optical "
                            "modules after the newest measured CCD are trainable"
                        )
                    ),
                }
                torch.save(payload, output)
        load_checkpoint(output, replacement, readout)
        query_embeddings = _hardware_embeddings(
            loaded,
            replacement,
            readout,
            settings,
            session_dir,
            list(bundle.test_samples),
            measured_stages,
        )
        gallery_embeddings = _hardware_embeddings(
            loaded,
            replacement,
            readout,
            settings,
            session_dir,
            list(bundle.gallery_samples),
            measured_stages,
        )
        evaluation = evaluate_embeddings(
            query_embeddings,
            bundle.test_samples,
            gallery_embeddings,
            bundle.gallery_samples,
            bundle.class_names,
            settings.gallery_aggregation,
            system_name=f"hardware_through_{stage}",
        )
        write_json(
            _stage_dir(session_dir, stage) / "finetune_metrics.json",
            {
                **evaluation.metrics,
                "stage": stage,
                "checkpoint": str(output),
                "measured_stages": list(measured_stages),
                "upstream_source": upstream_source,
                "background_subtraction": False,
            },
        )
        print(
            f"[finetune_{stage}] test_top1="
            f"{evaluation.metrics['top1_retrieval_accuracy']:.4f}",
            flush=True,
        )
    finally:
        _clear_measurements(replacement)
        replacement.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compact four-stage optical export and downstream fine-tuning"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--phase", choices=("export", "finetune"), required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--upstream-source",
        choices=UPSTREAM_SOURCES,
        default="measured",
        help=(
            "measured requires the preceding stage CCD folders; simulation "
            "runs all preceding optical stages in the trained simulator and "
            "is the fast single-stage hardware path"
        ),
    )
    args = parser.parse_args()
    settings = load_settings(args.config)
    seed_everything(settings.random_seed)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    session_dir = Path(args.session_dir).expanduser().resolve()
    if args.phase == "export":
        export_stage(
            settings,
            checkpoint,
            session_dir,
            args.stage,
            upstream_source=args.upstream_source,
        )
    else:
        finetune_stage(
            settings,
            checkpoint,
            session_dir,
            args.stage,
            args.epochs,
            upstream_source=args.upstream_source,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
