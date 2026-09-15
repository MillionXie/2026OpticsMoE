from __future__ import annotations

import argparse
import csv
import hashlib
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F

from experiments.hardware_sdk.workflows.reconstruct_slm import (
    encode_active_amplitude_with_metadata,
    encode_active_phase,
    reconstruct_directory,
    save_active_png,
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
from .offline_tail import LanguageGlobalOfflineTail
from .settings import load_settings


STAGES = (
    "vision_expert",
    "vision_global",
    "language_expert",
    "language_global",
)
UPSTREAM_SOURCES = ("measured", "simulation")
OFFLINE_SPLIT_CODES = {"train": 0, "gallery": 1, "test": 2}


def _load_measured_ccd_uint8(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        if image.mode != "L":
            raise RuntimeError(
                f"Measured CCD {path} must be an 8-bit grayscale image (mode L), "
                f"got {image.mode!r}"
            )
        array = np.asarray(image)
    if array.dtype != np.uint8:
        raise RuntimeError(f"Measured CCD {path} must use uint8 pixels")
    return torch.from_numpy(array.copy()).float()


def _load_hardware_ccd(
    stage_dir: Path, key: str, *, settings: Any
) -> torch.Tensor:
    root = stage_dir / "ccd_captured"
    candidates = [root / f"{key}{suffix}" for suffix in (".png", ".bmp", ".tif", ".tiff")]
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one measured CCD for {key} below {root}, found {len(matches)}"
        )
    value = _load_measured_ccd_uint8(matches[0])
    if bool(settings.hardware_ccd_flip_vertical):
        value = torch.flip(value, (-2,))
    if bool(settings.hardware_ccd_flip_horizontal):
        value = torch.flip(value, (-1,))
    target = int(settings.hardware_ccd_target_size)
    factor = int(settings.hardware_ccd_physical_binning_factor)
    if tuple(value.shape) == (target, target):
        pass
    elif tuple(value.shape) == (target * factor, target * factor):
        value = value.reshape(target, factor, target, factor).mean(dim=(1, 3))
    else:
        raise RuntimeError(
            f"CCD {matches[0]} is {tuple(value.shape)}; expected {target}x{target} "
            f"or {target * factor}x{target * factor}"
        )
    value = value.float().clamp_min(0.0)
    if not torch.isfinite(value).all():
        raise RuntimeError(f"CCD {matches[0]} contains invalid intensity")
    return value


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


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stage_dir(session_dir: Path, stage: str) -> Path:
    return session_dir / f"{STAGES.index(stage) + 1:02d}_{stage}"


def _samples(bundle: Any) -> list[Any]:
    return list(bundle.all_samples())


def _module_state_with_prefix(
    destination: dict[str, torch.Tensor], prefix: str, module: torch.nn.Module
) -> None:
    for name, value in module.state_dict().items():
        destination[f"{prefix}.{name}"] = value.detach().cpu().clone()


def _language_global_offline_tail_state(
    replacement: Any, readout: Any
) -> dict[str, torch.Tensor]:
    """Extract only the trainable tail after the frozen Language Block-1 boundary."""

    core = replacement.language_surrogate.core
    state: dict[str, torch.Tensor] = {}
    _module_state_with_prefix(state, "block2", core.blocks[1])
    _module_state_with_prefix(
        state, "ccd_readout", core.optical_branch.core.readout
    )
    _module_state_with_prefix(
        state,
        "optical_output_adapter",
        core.optical_branch.core.output_adapter,
    )
    _module_state_with_prefix(state, "output_norm", core.output_norm)
    state["block2_optical_fusion_logit"] = (
        core.block2_optical_fusion_logit.detach().cpu().clone()
    )
    _module_state_with_prefix(state, "retrieval_norm", readout.norm)
    _module_state_with_prefix(state, "retrieval_projection", readout.projection)
    return state


def _offline_tail_construction(settings: Any, replacement: Any) -> dict[str, Any]:
    core = replacement.language_surrogate.core
    block2 = core.blocks[1]
    optical = core.optical_branch
    detector_readout = optical.core.readout
    return {
        "width": int(core.width),
        "max_tokens": int(core.max_tokens),
        "expansion": float(block2.mlp[0].out_features / core.width),
        "dropout": float(block2.mlp[2].p),
        "initial_residual_weight": float(settings.electronic_initial_residual_weight),
        "token_mixer_enabled": bool(block2.token_mixer_enabled),
        "token_mixer_type": str(block2.token_mixer_type),
        "token_mixer_kernel_size": int(block2.token_mixer_kernel_size),
        "detector_size": int(optical.ccd_normalizer.active_size),
        "detector_output_size": int(detector_readout.output_size),
        "detector_layernorm_eps": float(detector_readout.norm.eps),
        "detector_layernorm_affine": bool(
            detector_readout.norm.elementwise_affine
        ),
        "detector_layernorm_scope": str(detector_readout.layernorm_scope),
        "detector_nonlinearity": str(detector_readout.nonlinearity),
        "ccd_relative_clip": float(optical.ccd_normalizer.relative_clip),
        "ccd_log_compression": float(optical.ccd_normalizer.log_compression),
        "minimum_optical_fusion": float(core.minimum_optical_fusion),
        "embedding_dim": int(settings.embedding_dim),
    }


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _write_language_global_offline_payload(
    *,
    settings: Any,
    replacement: Any,
    checkpoint: Path,
    session_dir: Path,
    destination: Path,
    rows: list[dict[str, Any]],
    block2_input_groups: list[torch.Tensor],
    tail_state: dict[str, torch.Tensor],
    upstream_source: str,
    measured_upstream_stages: tuple[str, ...],
) -> None:
    if len(rows) != len(block2_input_groups):
        raise RuntimeError(
            "Language-global offline cache count does not match the hardware manifest"
        )
    if not rows:
        raise RuntimeError("Cannot write an empty Language-global offline cache")
    required_splits = set(OFFLINE_SPLIT_CODES)
    seen_keys: set[str] = set()
    split_counts = {name: 0 for name in OFFLINE_SPLIT_CODES}
    class_names: dict[int, str] = {}
    class_split_counts: dict[int, dict[str, int]] = {}
    lengths: list[int] = []
    labels: list[int] = []
    split_codes: list[int] = []
    for index, (row, group) in enumerate(zip(rows, block2_input_groups)):
        key = str(row["key"])
        split = str(row["split"])
        label = int(row["sku_index"])
        name = str(row["sku_name"])
        if int(row["order"]) != index:
            raise RuntimeError("Hardware manifest order is not contiguous")
        if not key or Path(key).name != key or key in seen_keys:
            raise RuntimeError(f"Invalid or duplicate hardware key {key!r}")
        if split not in required_splits:
            raise RuntimeError(f"Unsupported offline split {split!r}")
        if group.ndim != 2 or group.shape[-1] != int(
            replacement.language_surrogate.core.width
        ):
            raise RuntimeError("Cached Language Block-2 input must be [L,width]")
        if len(group) <= 0 or len(group) > int(
            replacement.language_surrogate.core.max_tokens
        ):
            raise RuntimeError(f"Invalid cached Language token length {len(group)}")
        if not torch.isfinite(group).all():
            raise RuntimeError(f"Cached Language Block-2 input for {key} is non-finite")
        if label in class_names and class_names[label] != name:
            raise RuntimeError(f"Label {label} maps to multiple class names")
        seen_keys.add(key)
        class_names[label] = name
        class_split_counts.setdefault(
            label, {split_name: 0 for split_name in OFFLINE_SPLIT_CODES}
        )[split] += 1
        split_counts[split] += 1
        lengths.append(len(group))
        labels.append(label)
        split_codes.append(OFFLINE_SPLIT_CODES[split])
    expected_labels = list(range(len(class_names)))
    if sorted(class_names) != expected_labels:
        raise RuntimeError(
            f"Offline class indexes must be contiguous from zero, got {sorted(class_names)}"
        )
    if any(count <= 0 for count in split_counts.values()):
        raise RuntimeError(f"Every offline split must be present, got {split_counts}")
    quick210_requested = (
        getattr(settings, "gallery_images_per_sku", None) == 1
        and getattr(settings, "train_limit_per_sku", None) == 10
        and getattr(settings, "test_limit_per_sku", None) == 10
    )
    if quick210_requested:
        expected_per_class = {"train": 10, "gallery": 1, "test": 10}
        if (
            len(class_names) != 10
            or len(rows) != 210
            or split_counts != {"train": 100, "gallery": 10, "test": 100}
            or any(
                class_split_counts[index] != expected_per_class
                for index in expected_labels
            )
        ):
            raise RuntimeError(
                "quick210 requires exactly 10 classes with train/gallery/test "
                "counts 10/1/10 per class"
            )

    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    packed = torch.cat(
        [group.detach().cpu().float().contiguous() for group in block2_input_groups],
        dim=0,
    )
    cache = {
        "packed_block2_inputs": packed,
        "offsets": torch.tensor(offsets, dtype=torch.int64),
        "lengths": torch.tensor(lengths, dtype=torch.int64),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "split_codes": torch.tensor(split_codes, dtype=torch.uint8),
        "orders": torch.arange(len(rows), dtype=torch.int64),
    }
    construction = _offline_tail_construction(settings, replacement)
    nonfinite_state = [
        name
        for name, value in tail_state.items()
        if value.is_floating_point() and not torch.isfinite(value).all()
    ]
    if nonfinite_state:
        raise RuntimeError(
            f"Language-global offline tail state is non-finite: {nonfinite_state}"
        )
    validation_tail = LanguageGlobalOfflineTail(**construction)
    validation_tail.load_state_dict(tail_state, strict=True)
    tail_parameter_count = sum(
        parameter.numel() for parameter in validation_tail.parameters()
    )
    if quick210_requested and tail_parameter_count != 255_811:
        raise RuntimeError(
            "Formal quick210 Language-global tail must contain exactly 255,811 "
            f"parameters, got {tail_parameter_count}"
        )
    offline_dir = destination / "offline_downstream"
    cache_path = offline_dir / "cache.pt"
    state_path = offline_dir / "downstream_state.pt"
    _atomic_torch_save(cache, cache_path)
    _atomic_torch_save(tail_state, state_path)
    manifest_path = session_dir / "manifest.csv"
    ordered_keys = [str(row["key"]) for row in rows]
    contract = {
        "schema_version": 1,
        "type": "language_global_quick_offline_full_parity",
        "profile": "quick210" if quick210_requested else "generic",
        "stage": "language_global",
        "checkpoint_architecture": str(replacement.checkpoint_architecture),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": _sha256(checkpoint),
        "upstream_source": upstream_source,
        "measured_upstream_stages": list(measured_upstream_stages),
        "sample_count": len(rows),
        "manifest_relative_path": "../../manifest.csv",
        "manifest_sha256": _sha256(manifest_path),
        "ordered_keys_sha256": _sha256_text("\n".join(ordered_keys)),
        "cache_file": cache_path.name,
        "cache_sha256": _sha256(cache_path),
        "state_file": state_path.name,
        "state_sha256": _sha256(state_path),
        "cache_dtype": "float32",
        "cache_tensor": "Language Block-1 fused latent / Block-2 input [L,192]",
        "tail_construction": construction,
        "tail_trainable_parameter_count": int(tail_parameter_count),
        "tail_state_source_mapping": {
            "block2": "language_optical.core.blocks.1",
            "ccd_readout": "language_optical.core.optical_branch.core.readout",
            "optical_output_adapter": (
                "language_optical.core.optical_branch.core.output_adapter"
            ),
            "output_norm": "language_optical.core.output_norm",
            "block2_optical_fusion_logit": (
                "language_optical.core.block2_optical_fusion_logit"
            ),
            "retrieval_norm": "retrieval_readout.norm",
            "retrieval_projection": "retrieval_readout.projection",
        },
        "split_codes": OFFLINE_SPLIT_CODES,
        "split_counts": split_counts,
        "class_names": [class_names[index] for index in expected_labels],
        "class_split_counts": {
            str(index): class_split_counts[index] for index in expected_labels
        },
        "ccd_contract": {
            "directory_relative_to_stage": "ccd_captured",
            "filename": "<manifest-key>.png",
            "mode": "L",
            "dtype": "uint8",
            "shape_hw": [
                int(settings.hardware_ccd_target_size),
                int(settings.hardware_ccd_target_size),
            ],
            "flip_vertical_after_load": bool(settings.hardware_ccd_flip_vertical),
            "flip_horizontal_after_load": bool(
                settings.hardware_ccd_flip_horizontal
            ),
            "background_subtraction": False,
            "resizing": False,
            "normalization_order": [
                "clamp_nonnegative",
                "divide_by_single_frame_mean",
                f"relative_clip_{construction['ccd_relative_clip']}",
                f"log1p_factor_{construction['ccd_log_compression']}",
                "adaptive_avg_pool_478_to_224",
                "per_token_layernorm",
                str(construction["detector_nonlinearity"]),
            ],
        },
        "training_contract": {
            "recommended_epochs": 100,
            "seed": int(settings.random_seed),
            "pk_classes": int(settings.pk_skus_per_batch),
            "pk_images_per_class": int(settings.pk_images_per_sku),
            "learning_rate": float(settings.learning_rate),
            "readout_learning_rate": float(settings.readout_learning_rate),
            "weight_decay": float(settings.weight_decay),
            "gradient_clip_norm": 1.0,
            "supervised_contrastive_temperature": float(settings.temperature),
            "episodic_prototype_temperature": float(settings.gallery_temperature),
            "loss": "supervised_contrastive + episodic_prototype",
            "checkpoint_selection": (
                "fixed_train_development_top1_then_ce; sealed_test_once_after_selection"
            ),
            "block2_mode": "eval_with_gradients_dropout_disabled",
            "gallery_aggregation": str(settings.gallery_aggregation),
            "token_pooling": "mean_max",
            "embedding_normalization": "l2",
        },
        "excluded_from_offline_tail": [
            "Qwen backbone",
            "Vision optics/electronics",
            "Language input adapter and Block 1",
            "all phase masks and routers",
            "simulation propagation kernels",
        ],
    }
    write_json(offline_dir / "contract.json", contract)


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
    return _load_hardware_ccd(_stage_dir(session_dir, stage), key, settings=settings)


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
    inference_batch_size: int = 10,
) -> None:
    if inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be positive")
    stage_index = STAGES.index(stage)
    measured_upstream_stages = _measurement_plan(
        stage, upstream_source, include_current=False
    )
    bundle = prepare_caltech101_subset(settings, persist=True)
    samples = _samples(bundle)
    loaded, replacement, readout = _load_model(settings, checkpoint)
    offline_tail_state = (
        _language_global_offline_tail_state(replacement, readout)
        if stage == "language_global"
        else None
    )
    offline_block2_inputs: list[torch.Tensor] = []
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
    else:
        existing_rows = _read_manifest(session_dir)
        identity_fields = ("order", "key", "split", "sku_index", "sku_name")
        existing_identity = [
            tuple(str(row[field]) for field in identity_fields)
            for row in existing_rows
        ]
        expected_identity = [
            tuple(str(row[field]) for field in identity_fields) for row in rows
        ]
        if existing_identity != expected_identity:
            raise RuntimeError(
                "Existing hardware manifest order/split/class metadata does not "
                "match this dataset"
            )
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
        for start in range(0, len(samples), inference_batch_size):
            batch = samples[start : start + inference_batch_size]
            keys = [_key(sample) for sample in batch]
            _install_measurements(
                replacement,
                settings,
                session_dir,
                keys,
                measured_stages=measured_upstream_stages,
            )
            _forward_samples(loaded, replacement, readout, settings, batch)
            if stage == "language_global":
                cached_groups = (
                    replacement.language_surrogate.core.last_block2_input_groups
                )
                if len(cached_groups) != len(batch):
                    raise RuntimeError(
                        "Language-global hook did not preserve the export batch layout"
                    )
                offline_block2_inputs.extend(
                    group.detach().cpu().float().clone() for group in cached_groups
                )
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
    if stage == "language_global":
        if offline_tail_state is None:
            raise RuntimeError("Language-global offline tail state was not captured")
        _write_language_global_offline_payload(
            settings=settings,
            replacement=replacement,
            checkpoint=checkpoint,
            session_dir=session_dir,
            destination=destination,
            rows=rows,
            block2_input_groups=offline_block2_inputs,
            tail_state=offline_tail_state,
            upstream_source=upstream_source,
            measured_upstream_stages=measured_upstream_stages,
        )
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
            "server_persistence": (
                "one packed float32 Language Block-2 input cache; no simulation CCD "
                "and no per-sample PT cache"
                if stage == "language_global"
                else "no simulation CCD and no per-sample float32 PT cache"
            ),
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
    batch_size: int = 10,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("Hardware embedding batch_size must be positive")
    chunks = []
    _set_replacement_eval(replacement)
    replacement.set_phase_dropout_active(False)
    readout.eval()
    for start in range(0, len(samples), batch_size):
        chunks.append(
            _batch_embeddings(
                loaded,
                replacement,
                readout,
                settings,
                session_dir,
                samples[start : start + batch_size],
                measured_stages,
            ).detach().cpu()
        )
    return torch.cat(chunks, dim=0)


def _split_hardware_development(
    grouped: dict[int, list[Any]], *, seed: int, development_per_class: int
) -> tuple[dict[int, list[Any]], list[Any], list[Any]]:
    """Make a fixed development support/query set from captured train images."""

    if development_per_class < 2:
        raise ValueError("development_per_class must be at least 2")
    generator = torch.Generator().manual_seed(int(seed))
    fitting: dict[int, list[Any]] = {}
    support: list[Any] = []
    query: list[Any] = []
    for label in sorted(grouped):
        values = grouped[label]
        if len(values) <= development_per_class:
            raise RuntimeError(
                f"Class {label} has {len(values)} train captures; development "
                f"selection requires more than {development_per_class}"
            )
        order = torch.randperm(len(values), generator=generator).tolist()
        held_out = [values[index] for index in order[:development_per_class]]
        fitting[label] = [values[index] for index in order[development_per_class:]]
        support.append(held_out[0])
        query.extend(held_out[1:])
    return fitting, support, query


@torch.no_grad()
def _hardware_development_metrics(
    loaded: Any,
    replacement: Any,
    readout: Any,
    settings: Any,
    session_dir: Path,
    support_samples: list[Any],
    query_samples: list[Any],
    measured_stages: tuple[str, ...],
    *,
    inference_batch_size: int,
) -> dict[str, float]:
    support = F.normalize(
        _hardware_embeddings(
            loaded,
            replacement,
            readout,
            settings,
            session_dir,
            support_samples,
            measured_stages,
            batch_size=inference_batch_size,
        ).float(),
        dim=-1,
    )
    query = F.normalize(
        _hardware_embeddings(
            loaded,
            replacement,
            readout,
            settings,
            session_dir,
            query_samples,
            measured_stages,
            batch_size=inference_batch_size,
        ).float(),
        dim=-1,
    )
    labels = sorted({int(sample.sku_index) for sample in support_samples})
    if len(labels) != len(support_samples):
        raise RuntimeError("Development support requires exactly one image per class")
    label_to_target = {label: index for index, label in enumerate(labels)}
    prototypes = torch.stack(
        [
            F.normalize(
                support[
                    next(
                        index
                        for index, sample in enumerate(support_samples)
                        if int(sample.sku_index) == label
                    )
                ],
                dim=0,
            )
            for label in labels
        ]
    )
    targets = torch.tensor(
        [label_to_target[int(sample.sku_index)] for sample in query_samples],
        dtype=torch.long,
    )
    logits = query @ prototypes.T / float(settings.gallery_temperature)
    return {
        "development_top1": float(logits.argmax(dim=1).eq(targets).float().mean()),
        "development_ce": float(F.cross_entropy(logits, targets)),
    }


def finetune_stage(
    settings: Any,
    checkpoint: Path,
    session_dir: Path,
    stage: str,
    epochs: int,
    upstream_source: str = "measured",
    selection_policy: str = "development",
    development_per_class: int = 2,
    pk_classes: int | None = None,
    pk_images_per_class: int | None = None,
    inference_batch_size: int = 10,
    early_stopping_patience: int = 0,
) -> None:
    if selection_policy not in {"development", "train_loss"}:
        raise ValueError("selection_policy must be development or train_loss")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be positive")
    if early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be nonnegative")
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
    development_support: list[Any] = []
    development_query: list[Any] = []
    if selection_policy == "development":
        grouped, development_support, development_query = _split_hardware_development(
            grouped,
            seed=settings.random_seed,
            development_per_class=development_per_class,
        )
    class_count = int(
        min(len(grouped), settings.pk_skus_per_batch)
        if pk_classes is None
        else pk_classes
    )
    images_per_class = int(
        settings.pk_images_per_sku
        if pk_images_per_class is None
        else pk_images_per_class
    )
    if not 2 <= class_count <= len(grouped):
        raise ValueError(f"pk_classes must be in [2,{len(grouped)}]")
    if images_per_class < 2 or any(
        len(values) < images_per_class for values in grouped.values()
    ):
        raise ValueError(
            "pk_images_per_class must be at least 2 and no larger than every "
            "class's fitting split"
        )
    generator = torch.Generator().manual_seed(settings.random_seed)
    best = float("inf")
    best_epoch = 0
    best_development_top1 = float("nan")
    best_development_ce = float("nan")
    best_score: tuple[float, float, float] | None = None
    epochs_without_improvement = 0
    log_rows: list[dict[str, Any]] = []
    output = session_dir / "checkpoints" / f"after_{stage}.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for epoch in range(1, epochs + 1):
            _set_hardware_finetune_mode(replacement, readout, stage)
            batch_size = class_count * images_per_class
            steps = max(
                1,
                math.ceil(sum(map(len, grouped.values())) / batch_size),
            )
            total = 0.0
            total_contrastive = 0.0
            total_prototype = 0.0
            total_router_balance = 0.0
            total_router_importance = 0.0
            for _ in range(steps):
                batch: list[Any] = []
                label_order = torch.randperm(len(grouped), generator=generator)
                selected_labels = [
                    sorted(grouped)[int(index)] for index in label_order[:class_count]
                ]
                for label in selected_labels:
                    indexes = torch.randperm(
                        len(grouped[label]), generator=generator
                    )[:images_per_class]
                    batch.extend(
                        grouped[label][int(index)]
                        for index in indexes[:images_per_class]
                    )
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
            development = (
                _hardware_development_metrics(
                    loaded,
                    replacement,
                    readout,
                    settings,
                    session_dir,
                    development_support,
                    development_query,
                    measured_stages,
                    inference_batch_size=inference_batch_size,
                )
                if selection_policy == "development"
                else {
                    "development_top1": float("nan"),
                    "development_ce": float("nan"),
                }
            )
            row = {
                "epoch": epoch,
                "train_loss": average,
                "supervised_contrastive": average_contrastive,
                "episodic_prototype": average_prototype,
                "router_balance": average_router_balance,
                "router_importance": average_router_importance,
                **development,
            }
            log_rows.append(row)
            print(
                f"[finetune_{stage}] epoch={epoch:03d}/{epochs:03d} "
                f"loss={average:.5f} supcon={average_contrastive:.5f} "
                f"prototype={average_prototype:.5f} "
                f"router_balance={average_router_balance:.5f} "
                f"router_importance={average_router_importance:.5f} "
                f"dev_top1={development['development_top1']:.4f} "
                f"dev_ce={development['development_ce']:.5f}",
                flush=True,
            )
            score = (
                (
                    development["development_top1"],
                    -development["development_ce"],
                    -average,
                )
                if selection_policy == "development"
                else (-average, 0.0, 0.0)
            )
            improved = best_score is None or score > best_score
            if improved:
                best = average
                best_epoch = epoch
                best_development_top1 = development["development_top1"]
                best_development_ce = development["development_ce"]
                best_score = score
                epochs_without_improvement = 0
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
                    "selection_policy": selection_policy,
                    "development_top1": best_development_top1,
                    "development_ce": best_development_ce,
                    "sealed_test_used_for_selection": False,
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
            else:
                epochs_without_improvement += 1
            if (
                early_stopping_patience > 0
                and epochs_without_improvement >= early_stopping_patience
            ):
                print(
                    f"[finetune_{stage}] early_stop epoch={epoch:03d} "
                    f"best_epoch={best_epoch:03d}",
                    flush=True,
                )
                break
        write_csv(
            _stage_dir(session_dir, stage) / "finetune_train_log.csv",
            log_rows,
            list(log_rows[0]),
        )
        load_checkpoint(output, replacement, readout)
        query_embeddings = _hardware_embeddings(
            loaded,
            replacement,
            readout,
            settings,
            session_dir,
            list(bundle.test_samples),
            measured_stages,
            batch_size=inference_batch_size,
        )
        gallery_embeddings = _hardware_embeddings(
            loaded,
            replacement,
            readout,
            settings,
            session_dir,
            list(bundle.gallery_samples),
            measured_stages,
            batch_size=inference_batch_size,
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
                "epochs_requested": epochs,
                "epochs_completed": len(log_rows),
                "best_epoch": best_epoch,
                "best_train_loss": best,
                "selection_policy": selection_policy,
                "best_development_top1": best_development_top1,
                "best_development_ce": best_development_ce,
                "development_images_per_class": (
                    development_per_class
                    if selection_policy == "development"
                    else 0
                ),
                "sealed_test_used_for_selection": False,
                "test_evaluations_during_selection": 0,
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
        "--selection-policy",
        choices=("development", "train_loss"),
        default="development",
        help=(
            "development selects checkpoints on a fixed held-out subset of the "
            "captured train split; the sealed test is evaluated only once"
        ),
    )
    parser.add_argument("--development-per-class", type=int, default=2)
    parser.add_argument("--pk-classes", type=int)
    parser.add_argument("--pk-images-per-class", type=int)
    parser.add_argument("--inference-batch-size", type=int, default=10)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="0 disables early stopping",
    )
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
            inference_batch_size=args.inference_batch_size,
        )
    else:
        finetune_stage(
            settings,
            checkpoint,
            session_dir,
            args.stage,
            args.epochs,
            upstream_source=args.upstream_source,
            selection_policy=args.selection_policy,
            development_per_class=args.development_per_class,
            pk_classes=args.pk_classes,
            pk_images_per_class=args.pk_images_per_class,
            inference_batch_size=args.inference_batch_size,
            early_stopping_patience=args.early_stopping_patience,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
