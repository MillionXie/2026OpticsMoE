from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, cast

import yaml

from FixedFeedbackSFT.paths import REPOSITORY_ROOT, resolve_repository_path


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPOSITORY_ROOT

TaskName = Literal["caltech101", "isic2016", "lsp"]
MethodName = Literal["noft", "bp", "fa_pretrained", "fa_random"]

TASKS: tuple[TaskName, ...] = ("caltech101", "isic2016", "lsp")
METHODS: tuple[MethodName, ...] = (
    "noft",
    "bp",
    "fa_pretrained",
    "fa_random",
)

IMPLEMENTATION_FILES: tuple[str, ...] = (
    "FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/datasets.py",
    "FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/model.py",
    "FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/settings.py",
    "FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/training.py",
    "FixedFeedbackSFT/projects/d2nn_cifar10_high_performance_optical_backbone/optics.py",
    "FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone/model.py",
    "FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_slim_mixer_imagenet_backbone/model.py",
    "FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/model.py",
    "FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/stem.py",
    "experiments/qwen3_vl_embedding_2b_caltech101_object10_optical_retrieval/prepare_caltech101_retrieval_subset.py",
    "experiments/qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation/datasets.py",
    "experiments/qwen3_vl_embedding_2b_isic2016_skin_lesion_optical_segmentation/metrics.py",
    "experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/datasets.py",
    "experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/losses.py",
    "experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/metrics.py",
    "experiments/qwen3_vl_embedding_2b_fss1000_vision_optical_saliency/objectives.py",
    "experiments/qwen3_vl_embedding_2b_coco_duts_vision_optical_moe16_pretrain/objectives.py",
)
_REORGANIZED_PROJECT_PREFIX = "FixedFeedbackSFT/projects/"
LEGACY_IMPLEMENTATION_FILES: tuple[str, ...] = tuple(
    f"experiments/{relative.removeprefix(_REORGANIZED_PROJECT_PREFIX)}"
    if relative.startswith(_REORGANIZED_PROJECT_PREFIX)
    else relative
    for relative in IMPLEMENTATION_FILES
)


def implementation_files_for_repository(
    repository_root: str | Path,
) -> tuple[str, ...]:
    """Select the new layout, or the immutable pre-reorganization layout.

    New runs always use :data:`IMPLEMENTATION_FILES`.  The legacy candidate is
    retained solely so a derived mechanism audit can reconstruct the exact
    implementation digest of an already completed, pinned P12 worktree.
    """

    root = Path(repository_root).expanduser().resolve()
    for relative_paths in (IMPLEMENTATION_FILES, LEGACY_IMPLEMENTATION_FILES):
        if all((root / relative).is_file() for relative in relative_paths):
            return relative_paths
    first_new = next(
        relative for relative in IMPLEMENTATION_FILES if not (root / relative).is_file()
    )
    first_legacy = next(
        relative
        for relative in LEGACY_IMPLEMENTATION_FILES
        if not (root / relative).is_file()
    )
    raise FileNotFoundError(
        "Repository contains neither a complete reorganized nor legacy P12 "
        "implementation; first missing paths: "
        f"{first_new!r}, {first_legacy!r}"
    )


def implementation_sha256() -> str:
    """Fingerprint every source file that defines P12 data, forward, loss or metric."""

    digest = hashlib.sha256()
    for relative in IMPLEMENTATION_FILES:
        path = REPO_ROOT / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()

_P11_AXIS_SCHEDULE = (
    "token",
    "channel",
    "token",
    "channel",
    "token",
    "channel",
    "token",
    "channel",
)
_P11_SIGNATURE = (11, 1, 2, 4)


@dataclass(frozen=True)
class ModelSettings:
    canvas_size: int
    optical_channels: int
    num_stages: int
    token_dim: int
    token_count: int
    mixer_width: int
    axis_schedule: tuple[str, ...]
    architecture_signature: tuple[int, ...]
    optical_gate_min: float
    minimum_optical_parameter_fraction: float
    expected_optical_parameters: int


@dataclass(frozen=True)
class OptimizerSettings:
    phase_learning_rate: float
    adapter_learning_rate: float
    residual_learning_rate: float
    head_learning_rate: float
    phase_weight_decay: float
    electronic_weight_decay: float
    betas: tuple[float, float]
    eps: float
    warmup_epochs: int
    minimum_learning_rate_ratio: float
    phase_gradient_clip_norm: float
    electronic_gradient_clip_norm: float


@dataclass(frozen=True)
class TrainingSettings:
    head_only_epochs: int
    adaptation_epochs: int
    use_amp: bool
    amp_initial_scale: int
    amp_growth_interval: int
    deterministic_feedback: bool
    restore_rng_on_resume: bool


@dataclass(frozen=True)
class DataLoaderSettings:
    persistent_workers: bool
    pin_memory: bool
    prefetch_factor: int


@dataclass(frozen=True)
class SaveSettings:
    evaluation_interval_epochs: int
    checkpoint_interval_epochs: int
    log_interval_batches: int
    diagnostic_epochs: tuple[int, ...]
    save_last_every_epoch: bool
    run_final_ablations: bool


@dataclass(frozen=True)
class RunLimits:
    max_train_samples: int | None = None
    max_validation_samples: int | None = None
    max_test_samples: int | None = None
    max_train_batches: int | None = None
    max_validation_batches: int | None = None
    max_test_batches: int | None = None


@dataclass(frozen=True)
class TaskSettings:
    name: TaskName
    kind: str
    data_root: Path
    primary_metric: str
    num_outputs: int
    train_batch_size: int
    evaluation_batch_size: int
    num_workers: int
    validation_fraction: float
    head_hidden_dim: int | None
    decoder_width: int | None
    output_size: int | None
    expected_train_samples: int | None
    expected_test_samples: int | None


@dataclass(frozen=True)
class RunPaths:
    source_backbone: Path
    source_backbone_sha256: str
    stem_checkpoint: Path
    output_root: Path
    run_dir: Path
    common_start_dir: Path

    @property
    def common_start_checkpoint(self) -> Path:
        return self.common_start_dir / "common_start.pt"


@dataclass(frozen=True)
class Settings:
    """One fully resolved P12 task/method/seed run.

    A single YAML file defines the shared experiment. ``load_settings`` applies
    only run-identity and smoke-limit overrides, so optimization and the fixed
    P11 architecture cannot silently differ between feedback methods.
    """

    config_path: Path
    repo_root: Path
    task: TaskName
    method: MethodName
    seed: int
    paths: RunPaths
    model: ModelSettings
    optimizer: OptimizerSettings
    training: TrainingSettings
    dataloader: DataLoaderSettings
    save: SaveSettings
    limits: RunLimits
    task_settings: TaskSettings

    @property
    def updates_backbone(self) -> bool:
        return self.method != "noft"

    @property
    def feedback_mode(self) -> str:
        return {
            "noft": "none",
            "bp": "bp",
            "fa_pretrained": "fa_pretrained",
            "fa_random": "fa_random",
        }[self.method]

    @property
    def run_epochs(self) -> int:
        if self.method == "noft":
            return self.training.head_only_epochs
        return self.training.adaptation_epochs

    @property
    def inherited_pipeline_epochs(self) -> int:
        """Total trained epochs represented by an endpoint.

        Updating methods inherit the 50-epoch common head-only endpoint and
        then adapt for another 50 epochs. NoFT is that common endpoint itself.
        """

        if self.method == "noft":
            return self.training.head_only_epochs
        return self.training.head_only_epochs + self.training.adaptation_epochs

    @property
    def output_dir(self) -> Path:
        return self.paths.run_dir

    @property
    def data_root(self) -> Path:
        return self.task_settings.data_root

    @property
    def train_batch_size(self) -> int:
        return self.task_settings.train_batch_size

    @property
    def evaluation_batch_size(self) -> int:
        return self.task_settings.evaluation_batch_size

    @property
    def num_workers(self) -> int:
        return self.task_settings.num_workers

    @property
    def p11_config(self) -> dict[str, Any]:
        """Exact constructor config for the completed P11 source backbone.

        Keeping this mapping here prevents a downstream trainer from depending
        on mutable defaults in the ImageNet implementation. The ImageNet
        classifier is constructed only to verify the exported checkpoint and
        is removed by ``P11DownstreamModel`` immediately after loading.
        """

        return {
            "canvas_size": self.model.canvas_size,
            "optical_channels": self.model.optical_channels,
            "num_stages": self.model.num_stages,
            "token_dim": self.model.token_dim,
            "num_classes": 1000,
            "head_hidden_dim": 448,
            "wavelength_m": 5.32e-7,
            "pixel_size_m": 1.6e-5,
            "propagation_distance_m": 0.05,
            "token_axis_propagation_distance_m": 0.05,
            "channel_axis_propagation_distance_m": 0.05,
            "phase_init_std": 0.10,
            "layernorm_eps": 1.0e-5,
            "optical_gate_init": 0.60,
            "optical_gate_min": self.model.optical_gate_min,
            "mixer_width": self.model.mixer_width,
            "mixer_expansion": 2.0,
            "mixer_kernel_size": 3,
            "mixer_dropout": 0.10,
            "mixer_spatial_gate_init": 0.10,
            "mixer_channel_gate_init": 0.10,
            "residual_scale_init": 0.10,
            "residual_scale_max": 0.25,
            "optical_parameter_fraction_scope": "backbone_excluding_task_head",
            "minimum_optical_parameter_fraction": (
                self.model.minimum_optical_parameter_fraction
            ),
            # P11 source-module initialization used seed 2026. Loading the
            # strict checkpoint overwrites learned tensors, but retaining this
            # value makes missing-tensor failures reproducible and auditable.
            "seed": 2026,
        }

    def validate_runtime_paths(self, *, require_data: bool = True) -> None:
        required = {
            "source backbone": self.paths.source_backbone,
            "Qwen stem checkpoint": self.paths.stem_checkpoint,
        }
        if require_data:
            required["task data root"] = self.task_settings.data_root
        missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
        if missing:
            raise FileNotFoundError("Required P12 assets are missing:\n" + "\n".join(missing))

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], _jsonable(asdict(self)))

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def load_settings(
    path: str | Path,
    *,
    task: str | None = None,
    method: str | None = None,
    seed: int | None = None,
    output_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    limits: RunLimits | Mapping[str, int | None] | None = None,
) -> Settings:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a YAML mapping")

    selection = _mapping(raw, "selection")
    selected_task = str(task if task is not None else selection.get("task", ""))
    selected_method = str(method if method is not None else selection.get("method", ""))
    selected_seed = int(seed if seed is not None else selection.get("seed", -1))
    if selected_task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {selected_task!r}")
    if selected_method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {selected_method!r}")
    if selected_seed < 0:
        raise ValueError("seed must be a non-negative integer")

    paths_raw = _mapping(raw, "paths")
    source_backbone = _required_repo_path(paths_raw, "source_backbone")
    source_backbone_sha256 = str(_required(paths_raw, "source_backbone_sha256")).lower()
    stem_checkpoint = _required_repo_path(paths_raw, "stem_checkpoint")
    resolved_output_root = (
        _required_repo_path(paths_raw, "output_root")
        if output_root is None
        else _resolve_repo_path(output_root)
    )
    data_roots = _mapping(paths_raw, "data_roots")
    data_root = _required_repo_path(data_roots, selected_task)
    common_start_dir = (
        resolved_output_root / selected_task / "common" / f"seed_{selected_seed}"
    ).resolve()
    default_run_dir = (
        resolved_output_root
        / selected_task
        / selected_method
        / f"seed_{selected_seed}"
    ).resolve()
    run_dir = (
        default_run_dir
        if output_dir is None
        else _resolve_repo_path(output_dir)
    )

    model_raw = _mapping(raw, "model")
    model = ModelSettings(
        canvas_size=_integer(model_raw, "canvas_size"),
        optical_channels=_integer(model_raw, "optical_channels"),
        num_stages=_integer(model_raw, "num_stages"),
        token_dim=_integer(model_raw, "token_dim"),
        token_count=_integer(model_raw, "token_count"),
        mixer_width=_integer(model_raw, "mixer_width"),
        axis_schedule=tuple(str(value) for value in _sequence(model_raw, "axis_schedule")),
        architecture_signature=tuple(
            int(value) for value in _sequence(model_raw, "architecture_signature")
        ),
        optical_gate_min=_number(model_raw, "optical_gate_min"),
        minimum_optical_parameter_fraction=_number(
            model_raw, "minimum_optical_parameter_fraction"
        ),
        expected_optical_parameters=_integer(model_raw, "expected_optical_parameters"),
    )

    optimizer_raw = _mapping(raw, "optimizer")
    betas = tuple(float(value) for value in _sequence(optimizer_raw, "betas"))
    if len(betas) != 2:
        raise ValueError("optimizer.betas must contain exactly two values")
    optimizer = OptimizerSettings(
        phase_learning_rate=_number(optimizer_raw, "phase_learning_rate"),
        adapter_learning_rate=_number(optimizer_raw, "adapter_learning_rate"),
        residual_learning_rate=_number(optimizer_raw, "residual_learning_rate"),
        head_learning_rate=_number(optimizer_raw, "head_learning_rate"),
        phase_weight_decay=_number(optimizer_raw, "phase_weight_decay"),
        electronic_weight_decay=_number(optimizer_raw, "electronic_weight_decay"),
        betas=cast(tuple[float, float], betas),
        eps=_number(optimizer_raw, "eps"),
        warmup_epochs=_integer(optimizer_raw, "warmup_epochs"),
        minimum_learning_rate_ratio=_number(
            optimizer_raw, "minimum_learning_rate_ratio"
        ),
        phase_gradient_clip_norm=_number(
            optimizer_raw, "phase_gradient_clip_norm"
        ),
        electronic_gradient_clip_norm=_number(
            optimizer_raw, "electronic_gradient_clip_norm"
        ),
    )

    training_raw = _mapping(raw, "training")
    training = TrainingSettings(
        head_only_epochs=_integer(training_raw, "head_only_epochs"),
        adaptation_epochs=_integer(training_raw, "adaptation_epochs"),
        use_amp=_boolean(training_raw, "use_amp"),
        amp_initial_scale=_integer(training_raw, "amp_initial_scale"),
        amp_growth_interval=_integer(training_raw, "amp_growth_interval"),
        deterministic_feedback=_boolean(training_raw, "deterministic_feedback"),
        restore_rng_on_resume=_boolean(training_raw, "restore_rng_on_resume"),
    )

    dataloader_raw = _mapping(raw, "dataloader")
    dataloader = DataLoaderSettings(
        persistent_workers=_boolean(dataloader_raw, "persistent_workers"),
        pin_memory=_boolean(dataloader_raw, "pin_memory"),
        prefetch_factor=_integer(dataloader_raw, "prefetch_factor"),
    )

    save_raw = _mapping(raw, "save")
    save = SaveSettings(
        evaluation_interval_epochs=_integer(save_raw, "evaluation_interval_epochs"),
        checkpoint_interval_epochs=_integer(save_raw, "checkpoint_interval_epochs"),
        log_interval_batches=_integer(save_raw, "log_interval_batches"),
        diagnostic_epochs=tuple(
            int(value) for value in _sequence(save_raw, "diagnostic_epochs")
        ),
        save_last_every_epoch=_boolean(save_raw, "save_last_every_epoch"),
        run_final_ablations=_boolean(save_raw, "run_final_ablations"),
    )

    limits_raw = _mapping(raw, "limits")
    run_limits = _parse_limits(limits_raw)
    if limits is not None:
        if isinstance(limits, RunLimits):
            run_limits = limits
        elif isinstance(limits, Mapping):
            unknown = set(limits) - set(RunLimits.__dataclass_fields__)
            if unknown:
                raise ValueError(f"Unknown limit override(s): {sorted(unknown)}")
            run_limits = replace(run_limits, **dict(limits))
        else:
            raise TypeError("limits must be RunLimits, a mapping, or None")

    tasks_raw = _mapping(raw, "tasks")
    task_raw = _mapping(tasks_raw, selected_task)
    decoder_value = task_raw.get("decoder_width")
    output_size_value = task_raw.get("output_size")
    task_settings = TaskSettings(
        name=cast(TaskName, selected_task),
        kind=str(_required(task_raw, "kind")),
        data_root=data_root,
        primary_metric=str(_required(task_raw, "primary_metric")),
        num_outputs=_integer(task_raw, "num_outputs"),
        train_batch_size=_integer(task_raw, "train_batch_size"),
        evaluation_batch_size=_integer(task_raw, "evaluation_batch_size"),
        num_workers=_integer(task_raw, "num_workers"),
        validation_fraction=_number(task_raw, "validation_fraction"),
        head_hidden_dim=_optional_integer(task_raw.get("head_hidden_dim")),
        decoder_width=None if decoder_value is None else int(decoder_value),
        output_size=None if output_size_value is None else int(output_size_value),
        expected_train_samples=_optional_integer(task_raw.get("expected_train_samples")),
        expected_test_samples=_optional_integer(task_raw.get("expected_test_samples")),
    )

    settings = Settings(
        config_path=config_path,
        repo_root=REPO_ROOT,
        task=cast(TaskName, selected_task),
        method=cast(MethodName, selected_method),
        seed=selected_seed,
        paths=RunPaths(
            source_backbone=source_backbone,
            source_backbone_sha256=source_backbone_sha256,
            stem_checkpoint=stem_checkpoint,
            output_root=resolved_output_root,
            run_dir=run_dir,
            common_start_dir=common_start_dir,
        ),
        model=model,
        optimizer=optimizer,
        training=training,
        dataloader=dataloader,
        save=save,
        limits=run_limits,
        task_settings=task_settings,
    )
    _validate(settings)
    return settings


def add_settings_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--max-test-batches", type=int)
    return parser


def load_settings_from_args(args: argparse.Namespace) -> Settings:
    limit_values = {
        name: getattr(args, name, None)
        for name in RunLimits.__dataclass_fields__
        if getattr(args, name, None) is not None
    }
    return load_settings(
        args.config,
        task=getattr(args, "task", None),
        method=getattr(args, "method", None),
        seed=getattr(args, "seed", None),
        output_root=getattr(args, "output_root", None),
        output_dir=getattr(args, "output_dir", None),
        limits=limit_values or None,
    )


def _validate(settings: Settings) -> None:
    model = settings.model
    locked = {
        "canvas_size": (model.canvas_size, 224),
        "optical_channels": (model.optical_channels, 3),
        "num_stages": (model.num_stages, 8),
        "token_dim": (model.token_dim, 224),
        "token_count": (model.token_count, 196),
        "mixer_width": (model.mixer_width, 96),
        "axis_schedule": (model.axis_schedule, _P11_AXIS_SCHEDULE),
        "architecture_signature": (model.architecture_signature, _P11_SIGNATURE),
        "expected_optical_parameters": (model.expected_optical_parameters, 1_204_224),
    }
    changed = [
        f"{name}={actual!r} (required {expected!r})"
        for name, (actual, expected) in locked.items()
        if actual != expected
    ]
    if changed:
        raise ValueError("P12 must retain the fixed P11 architecture: " + "; ".join(changed))
    if model.optical_gate_min != 0.5:
        raise ValueError("P11 optical_gate_min must remain exactly 0.5")
    if model.minimum_optical_parameter_fraction < 0.5:
        raise ValueError("minimum_optical_parameter_fraction must be at least 0.5")
    expected_source_sha = settings.paths.source_backbone_sha256
    if len(expected_source_sha) != 64 or any(
        character not in "0123456789abcdef" for character in expected_source_sha
    ):
        raise ValueError("paths.source_backbone_sha256 must be a lowercase SHA-256 digest")

    training = settings.training
    if training.head_only_epochs != 50 or training.adaptation_epochs != 50:
        raise ValueError(
            "P12 locks head_only_epochs=50 and adaptation_epochs=50 for every task"
        )
    if not training.deterministic_feedback or not training.restore_rng_on_resume:
        raise ValueError("Formal P12 runs require deterministic feedback and RNG resume")
    if training.amp_initial_scale <= 0 or training.amp_growth_interval <= 0:
        raise ValueError("AMP scale and growth interval must be positive")

    optimizer = settings.optimizer
    if not 1.0e-3 <= optimizer.phase_learning_rate <= 1.0e-2:
        raise ValueError(
            "phase_learning_rate must remain in [1e-3,1e-2]; "
            "P12 uses 3e-3 and permits the preregistered 7e-3 screen"
        )
    for name in (
        "adapter_learning_rate",
        "residual_learning_rate",
        "head_learning_rate",
        "eps",
        "phase_gradient_clip_norm",
        "electronic_gradient_clip_norm",
    ):
        if float(getattr(optimizer, name)) <= 0:
            raise ValueError(f"optimizer.{name} must be positive")
    if optimizer.phase_weight_decay != 0.0:
        raise ValueError("Optical phase parameters must use zero weight decay")
    if optimizer.electronic_weight_decay < 0:
        raise ValueError("optimizer.electronic_weight_decay cannot be negative")
    if not all(0.0 <= beta < 1.0 for beta in optimizer.betas):
        raise ValueError("optimizer.betas must lie in [0,1)")
    if not 0 <= optimizer.warmup_epochs < training.adaptation_epochs:
        raise ValueError("optimizer.warmup_epochs must be in [0, adaptation_epochs)")
    if not 0.0 <= optimizer.minimum_learning_rate_ratio <= 1.0:
        raise ValueError("optimizer.minimum_learning_rate_ratio must be in [0,1]")

    task = settings.task_settings
    expected_kind = {
        "caltech101": "classification",
        "isic2016": "segmentation",
        "lsp": "pose",
    }[settings.task]
    expected_outputs = {"caltech101": 101, "isic2016": 1, "lsp": 14}[settings.task]
    if task.kind != expected_kind or task.num_outputs != expected_outputs:
        raise ValueError(
            f"{settings.task} requires kind={expected_kind!r} and "
            f"num_outputs={expected_outputs}"
        )
    for name in ("train_batch_size", "evaluation_batch_size"):
        if int(getattr(task, name)) <= 0:
            raise ValueError(f"tasks.{settings.task}.{name} must be positive")
    if task.kind == "classification" and (
        task.head_hidden_dim is None or task.head_hidden_dim <= 0
    ):
        raise ValueError("Caltech classification requires a positive head_hidden_dim")
    if task.num_workers < 0:
        raise ValueError(f"tasks.{settings.task}.num_workers cannot be negative")
    if not 0.0 < task.validation_fraction < 0.5:
        raise ValueError("task validation_fraction must lie in (0,0.5)")
    if task.kind in {"segmentation", "pose"}:
        if task.decoder_width is None or task.decoder_width <= 0:
            raise ValueError(f"{settings.task} requires a positive decoder_width")
        expected_output_size = 224 if settings.task == "isic2016" else 56
        if task.output_size != expected_output_size:
            raise ValueError(
                f"{settings.task} output_size must remain {expected_output_size}"
            )

    if settings.dataloader.prefetch_factor <= 0:
        raise ValueError("dataloader.prefetch_factor must be positive")
    save = settings.save
    if save.evaluation_interval_epochs != 1:
        raise ValueError("P12 validates every epoch; evaluation_interval_epochs must be 1")
    for name in (
        "evaluation_interval_epochs",
        "checkpoint_interval_epochs",
        "log_interval_batches",
    ):
        if int(getattr(save, name)) <= 0:
            raise ValueError(f"save.{name} must be positive")
    if not save.save_last_every_epoch:
        raise ValueError("Formal P12 resume requires save_last_every_epoch=true")
    if not save.diagnostic_epochs:
        raise ValueError("save.diagnostic_epochs cannot be empty")
    if tuple(sorted(set(save.diagnostic_epochs))) != save.diagnostic_epochs:
        raise ValueError("save.diagnostic_epochs must be sorted and unique")
    if save.diagnostic_epochs[0] != 0 or save.diagnostic_epochs[-1] != 50:
        raise ValueError("save.diagnostic_epochs must include endpoints 0 and 50")
    if any(epoch < 0 or epoch > 50 for epoch in save.diagnostic_epochs):
        raise ValueError("save.diagnostic_epochs must lie in [0,50]")

    for name, value in asdict(settings.limits).items():
        if value is not None and int(value) <= 0:
            raise ValueError(f"limits.{name} must be positive or null")


def _parse_limits(raw: Mapping[str, Any]) -> RunLimits:
    return RunLimits(
        **{
            name: _optional_integer(raw.get(name))
            for name in RunLimits.__dataclass_fields__
        }
    )


def _required(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required configuration key: {key}")
    return mapping[key]


def _mapping(mapping: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = _required(mapping, key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a YAML mapping")
    return value


def _sequence(mapping: Mapping[str, Any], key: str) -> Sequence[Any]:
    value = _required(mapping, key)
    if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes)):
        raise ValueError(f"{key} must be a YAML sequence")
    return value


def _integer(mapping: Mapping[str, Any], key: str) -> int:
    value = _required(mapping, key)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return int(value)


def _optional_integer(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("Optional integer values cannot be booleans")
    return int(value)


def _number(mapping: Mapping[str, Any], key: str) -> float:
    value = _required(mapping, key)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _boolean(mapping: Mapping[str, Any], key: str) -> bool:
    value = _required(mapping, key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be true or false")
    return value


def _required_repo_path(mapping: Mapping[str, Any], key: str) -> Path:
    value = _required(mapping, key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"{key} must be a non-empty path")
    return _resolve_repo_path(value)


def _resolve_repo_path(value: str | Path) -> Path:
    return resolve_repository_path(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


__all__ = [
    "DataLoaderSettings",
    "EXPERIMENT_DIR",
    "METHODS",
    "MethodName",
    "ModelSettings",
    "OptimizerSettings",
    "REPO_ROOT",
    "RunLimits",
    "RunPaths",
    "SaveSettings",
    "Settings",
    "TASKS",
    "TaskName",
    "TaskSettings",
    "TrainingSettings",
    "add_settings_arguments",
    "load_settings",
    "load_settings_from_args",
]
