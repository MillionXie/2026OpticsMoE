from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch
import yaml
from torch import nn
from torch.nn import functional as F

from . import training
from .model import P11DownstreamModel
from .settings import (
    METHODS,
    REPO_ROOT,
    RunLimits,
    Settings,
    implementation_sha256,
    load_settings,
)


PANEL_FORMAT = "p12-phase-only-adaptation-panel-v1"
ADAPTATION_SCOPE = "phase_and_head"
NOFT_SCOPE = "head_only"
PANEL_RUNTIME_FILES = (
    "experiments/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/phase_only.py",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def panel_implementation_sha256() -> str:
    """Hash the add-on policy without changing P12's locked base digest.

    The original P12 implementation digest is deliberately retained so that
    an e305 NoFT/common-start checkpoint remains loadable.  This second digest
    is embedded in the resolved settings and every phase-only result, making
    the add-on runtime independently identifiable.
    """

    digest = hashlib.sha256()
    digest.update(b"p12-phase-only-runtime\0")
    digest.update(implementation_sha256().encode("ascii"))
    for relative in PANEL_RUNTIME_FILES:
        path = REPO_ROOT / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class PhaseOnlyProtocol:
    panel_config: Path
    panel_config_sha256: str
    base_config: Path
    output_root: Path
    adaptation_scope: str
    noft_scope: str
    frozen_electronic_backbone_mode: str
    task_head_gradient: str

    def to_dict(self, *, method: str) -> dict[str, Any]:
        return {
            "format": PANEL_FORMAT,
            "panel_config": str(self.panel_config),
            "panel_config_sha256": self.panel_config_sha256,
            "base_config": str(self.base_config),
            "base_implementation_sha256": implementation_sha256(),
            "panel_implementation_sha256": panel_implementation_sha256(),
            "adaptation_scope": self.adaptation_scope,
            "noft_scope": self.noft_scope,
            "effective_scope": self.noft_scope if method == "noft" else self.adaptation_scope,
            "frozen_electronic_backbone_mode": self.frozen_electronic_backbone_mode,
            "task_head_gradient": self.task_head_gradient,
            "feedback_semantics": {
                "noft": "no optical update; exact-BP task head only",
                "bp": "current optical BP; phase and exact-BP task head update",
                "fa_pretrained": (
                    "source/pretrained fixed inter-stage optical connector; "
                    "phase and exact-BP task head update"
                ),
                "fa_random": (
                    "random fixed inter-stage optical connector; phase and "
                    "exact-BP task head update"
                ),
            },
        }


@dataclass(frozen=True)
class PhaseOnlySettings:
    """Transparent settings view that adds an identity-bearing panel policy."""

    base: Settings
    protocol: PhaseOnlyProtocol

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def to_dict(self) -> dict[str, Any]:
        payload = self.base.to_dict()
        payload["config_path"] = str(self.protocol.panel_config)
        payload["phase_only_panel"] = self.protocol.to_dict(method=self.base.method)
        return payload

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def _required_mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a YAML mapping")
    return value


def _repo_path(value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty repository-relative path")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def load_phase_only_settings(
    panel_config: str | Path,
    *,
    task: str | None = None,
    method: str | None = None,
    seed: int | None = None,
    output_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    limits: RunLimits | Mapping[str, int | None] | None = None,
    adaptation_scope: str | None = None,
) -> PhaseOnlySettings:
    panel_path = Path(panel_config).expanduser().resolve()
    raw = yaml.safe_load(panel_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Phase-only configuration root must be a YAML mapping")
    if raw.get("format") != PANEL_FORMAT:
        raise ValueError(f"phase-only config format must be {PANEL_FORMAT!r}")
    allowed = {"format", "base_config", "output_root", "phase_only_panel"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown phase-only config keys: {sorted(unknown)}")

    base_config = _repo_path(raw.get("base_config"), name="base_config")
    configured_output_root = _repo_path(raw.get("output_root"), name="output_root")
    policy = _required_mapping(raw, "phase_only_panel")
    expected_policy = {
        "adaptation_scope": ADAPTATION_SCOPE,
        "noft_scope": NOFT_SCOPE,
        "frozen_electronic_backbone_mode": "eval",
        "task_head_gradient": "exact_bp",
    }
    if set(policy) != set(expected_policy):
        raise ValueError(
            "phase_only_panel must contain exactly "
            f"{sorted(expected_policy)}; got {sorted(policy)}"
        )
    mismatches = {
        key: (policy.get(key), expected)
        for key, expected in expected_policy.items()
        if policy.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Unsupported phase-only policy: {mismatches}")
    requested_scope = adaptation_scope or str(policy["adaptation_scope"])
    if requested_scope != policy["adaptation_scope"]:
        raise ValueError(
            "CLI adaptation scope must match the locked panel config: "
            f"cli={requested_scope!r}, config={policy['adaptation_scope']!r}"
        )
    resolved_output_root = configured_output_root if output_root is None else output_root
    base = load_settings(
        base_config,
        task=task,
        method=method,
        seed=seed,
        output_root=resolved_output_root,
        output_dir=output_dir,
        limits=limits,
    )
    if base.paths.output_root == load_settings(base_config).paths.output_root:
        raise ValueError("Phase-only panel must use an output root isolated from joint P12")
    protocol = PhaseOnlyProtocol(
        panel_config=panel_path,
        panel_config_sha256=_sha256_file(panel_path),
        base_config=base_config,
        output_root=base.paths.output_root,
        adaptation_scope=str(policy["adaptation_scope"]),
        noft_scope=str(policy["noft_scope"]),
        frozen_electronic_backbone_mode=str(policy["frozen_electronic_backbone_mode"]),
        task_head_gradient=str(policy["task_head_gradient"]),
    )
    return PhaseOnlySettings(base=base, protocol=protocol)


def add_phase_only_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", type=Path, required=True, help="phase-only panel YAML")
    parser.add_argument("--task", choices=("caltech101", "isic2016", "lsp"))
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--adaptation-scope",
        choices=(ADAPTATION_SCOPE,),
        default=ADAPTATION_SCOPE,
        help="locked panel scope (explicitly passed by the command wrapper)",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--max-test-batches", type=int)
    return parser


def load_phase_only_settings_from_args(args: argparse.Namespace) -> PhaseOnlySettings:
    limit_values = {
        name: getattr(args, name, None)
        for name in RunLimits.__dataclass_fields__
        if getattr(args, name, None) is not None
    }
    return load_phase_only_settings(
        args.config,
        task=getattr(args, "task", None),
        method=getattr(args, "method", None),
        seed=getattr(args, "seed", None),
        output_root=getattr(args, "output_root", None),
        output_dir=getattr(args, "output_dir", None),
        limits=limit_values or None,
        adaptation_scope=getattr(args, "adaptation_scope", None),
    )


def _unique(parameters: Iterable[nn.Parameter]) -> tuple[nn.Parameter, ...]:
    values: list[nn.Parameter] = []
    seen: set[int] = set()
    for parameter in parameters:
        if id(parameter) not in seen:
            values.append(parameter)
            seen.add(id(parameter))
    return tuple(values)


def parameter_family_report(model: P11DownstreamModel) -> dict[str, Any]:
    named = dict(model.named_parameters())
    name_by_id = {id(parameter): name for name, parameter in named.items()}
    families = {
        "phase": _unique(model.phase_parameters()),
        "adapter": _unique(model.adapter_parameters()),
        "residual": _unique(model.residual_parameters()),
        "stem": _unique(model.backbone.stem.parameters()),
        "head": _unique(model.head_parameters()),
    }
    ids = {name: {id(parameter) for parameter in values} for name, values in families.items()}
    overlaps: dict[str, list[str]] = {}
    keys = tuple(families)
    for index, left in enumerate(keys):
        for right in keys[index + 1 :]:
            shared = ids[left] & ids[right]
            if shared:
                overlaps[f"{left}:{right}"] = sorted(name_by_id[value] for value in shared)
    if overlaps:
        raise RuntimeError(f"Parameter families overlap: {overlaps}")
    classified = set().union(*ids.values())
    unclassified = sorted(
        name for name, parameter in named.items() if id(parameter) not in classified
    )
    if unclassified:
        raise RuntimeError(f"Unclassified P12 parameters: {unclassified}")
    return {
        family: {
            "tensor_count": len(values),
            "parameter_count": sum(parameter.numel() for parameter in values),
            "names": [name_by_id[id(parameter)] for parameter in values],
            "trainable_tensor_count": sum(parameter.requires_grad for parameter in values),
        }
        for family, values in families.items()
    }


def set_phase_only_backbone_trainable(
    model: P11DownstreamModel, enabled: bool
) -> None:
    """Freeze every parameter, then opt in only the allowed two families."""

    model.requires_grad_(False)
    model.head.requires_grad_(True)
    if enabled:
        for parameter in model.phase_parameters():
            parameter.requires_grad_(True)


def _assert_trainable_policy(
    model: P11DownstreamModel, settings: PhaseOnlySettings
) -> dict[str, Any]:
    report = parameter_family_report(model)
    expected_phase_tensors = settings.model.num_stages
    if report["phase"]["tensor_count"] != expected_phase_tensors:
        raise RuntimeError(
            f"Expected {expected_phase_tensors} phase tensors, got "
            f"{report['phase']['tensor_count']}"
        )
    if report["phase"]["parameter_count"] != settings.model.expected_optical_parameters:
        raise RuntimeError("Phase-only panel optical parameter count changed")
    expected_trainable = {
        "phase": expected_phase_tensors if settings.updates_backbone else 0,
        "adapter": 0,
        "residual": 0,
        "stem": 0,
        "head": report["head"]["tensor_count"],
    }
    actual = {name: row["trainable_tensor_count"] for name, row in report.items()}
    if actual != expected_trainable:
        raise RuntimeError(
            "Phase-only trainability policy violation: "
            f"actual={actual}, expected={expected_trainable}"
        )
    return report


def phase_only_trainable_groups(
    model: P11DownstreamModel, settings: PhaseOnlySettings
) -> list[dict[str, Any]]:
    report = _assert_trainable_policy(model, settings)
    groups: list[dict[str, Any]] = []
    if settings.updates_backbone:
        groups.append(
            {
                "params": [parameter for parameter in model.phase_parameters()],
                "lr": settings.optimizer.phase_learning_rate,
                "weight_decay": settings.optimizer.phase_weight_decay,
                "group_name": "phase",
            }
        )
    groups.append(
        {
            "params": [parameter for parameter in model.head_parameters()],
            "lr": settings.optimizer.head_learning_rate,
            "weight_decay": settings.optimizer.electronic_weight_decay,
            "group_name": "head",
        }
    )
    optimizer_ids = {id(parameter) for group in groups for parameter in group["params"]}
    required_ids = {
        id(parameter)
        for parameter in (
            *(() if not settings.updates_backbone else tuple(model.phase_parameters())),
            *tuple(model.head_parameters()),
        )
    }
    if optimizer_ids != required_ids:
        raise RuntimeError("Optimizer parameters differ from phase/head policy")
    manifest = {
        **settings.protocol.to_dict(method=settings.method),
        "task": settings.task,
        "method": settings.method,
        "seed": settings.seed,
        "parameter_families": report,
        "optimizer_groups": [group["group_name"] for group in groups],
        "electronic_backbone_parameter_count": (
            report["adapter"]["parameter_count"] + report["residual"]["parameter_count"]
        ),
        "electronic_backbone_trainable_parameters": 0,
    }
    training.write_json(settings.output_dir / "phase_only_panel.json", manifest)
    return groups


def _head_gradients(model: P11DownstreamModel) -> tuple[torch.Tensor, ...]:
    values: list[torch.Tensor] = []
    for index, parameter in enumerate(model.head_parameters(), start=1):
        if parameter.grad is None:
            raise RuntimeError(f"Missing task-head gradient tensor {index}")
        gradient = parameter.grad.detach().float().cpu().clone()
        if not bool(torch.isfinite(gradient).all()):
            raise RuntimeError(f"Non-finite task-head gradient tensor {index}")
        values.append(gradient)
    if not values:
        raise RuntimeError("Task head contains no parameters")
    return tuple(values)


def _assert_frozen_electronics_have_no_grad(model: P11DownstreamModel) -> dict[str, int]:
    groups = {
        "adapter": tuple(model.adapter_parameters()),
        "residual": tuple(model.residual_parameters()),
        "stem": tuple(model.backbone.stem.parameters()),
    }
    violations = {
        name: sum(parameter.requires_grad or parameter.grad is not None for parameter in values)
        for name, values in groups.items()
    }
    if any(violations.values()):
        raise RuntimeError(f"Frozen electronics acquired trainability/gradients: {violations}")
    return {name: sum(parameter.numel() for parameter in values) for name, values in groups.items()}


def phase_only_gradient_diagnostic(
    model: P11DownstreamModel,
    raw_batch: Mapping[str, Any],
    settings: PhaseOnlySettings,
    device: torch.device,
) -> dict[str, Any] | None:
    if settings.method == "noft":
        return None
    was_training = model.training
    model.eval()
    batch = training._move_batch(raw_batch, device)
    for key, value in list(batch.items()):
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            batch[key] = value[:2]
        elif isinstance(value, list):
            batch[key] = value[:2]
    rng = training.capture_rng_state()

    def compute(method: str):
        training.restore_rng_state(rng)
        model.zero_grad(set_to_none=True)
        model.configure_feedback(method, random_seed=int(settings.seed) + 8_000_003)
        output = model(batch["image"])
        loss, _ = training.task_loss(settings, output, batch)
        loss.backward()
        phase = training._gradient_list(model)
        head = _head_gradients(model)
        head_report = training._parameter_gradient_report(
            model.head_parameters(), name="task_head"
        )
        frozen = _assert_frozen_electronics_have_no_grad(model)
        return phase, head, head_report, frozen

    exact_phase, exact_head, _, _ = compute("bp_current")
    candidate_phase, candidate_head, head_report, frozen = compute(
        training._feedback_method(settings)
    )
    phase_rows = training._gradient_comparison(exact_phase, candidate_phase)
    head_rows = []
    for index, (exact, candidate) in enumerate(
        zip(exact_head, candidate_head, strict=True), start=1
    ):
        maximum = float((exact - candidate).abs().max())
        cosine = float(
            F.cosine_similarity(
                exact.flatten().double(), candidate.flatten().double(), dim=0, eps=1.0e-20
            )
        )
        head_rows.append(
            {"tensor": index, "max_absolute_difference": maximum, "cosine": cosine}
        )
    max_head_difference = max(row["max_absolute_difference"] for row in head_rows)
    if max_head_difference > 1.0e-7:
        raise RuntimeError(
            "Task-head gradient changed between BP and optical-FA connector: "
            f"max_abs={max_head_difference:.3e}"
        )
    model.zero_grad(set_to_none=True)
    training.configure_runtime_feedback(model, settings)
    training.restore_rng_state(rng)
    model.train(was_training)
    if settings.method == "fa_pretrained" and model.phase_report()["mean_absolute_rad"] < 1.0e-7:
        minimum = min(float(row["cosine_to_bp_current"]) for row in phase_rows)
        if minimum < 0.999:
            raise RuntimeError(
                "FA-pretrained must match current BP at the unchanged common start; "
                f"minimum phase-gradient cosine was {minimum:.6f}"
            )
    return {
        "method": settings.method,
        "adaptation_scope": ADAPTATION_SCOPE,
        "phase_motion_from_source": model.phase_report(),
        "per_stage": phase_rows,
        "mean_cosine_stages_1_to_7": sum(
            float(row["cosine_to_bp_current"]) for row in phase_rows[:-1]
        )
        / max(len(phase_rows) - 1, 1),
        "last_stage_expected_exact_local_gradient": True,
        "task_head_gradient_is_exact_bp": True,
        "task_head_bp_vs_connector": {
            "maximum_absolute_difference": max_head_difference,
            "per_tensor": head_rows,
        },
        "frozen_electronic_backbone": frozen,
        "trainable_gradient_groups": {
            "phase": {"parameter_tensors": len(candidate_phase)},
            "task_head": head_report,
        },
    }


@contextmanager
def phase_only_runtime() -> Iterator[None]:
    """Patch only this process; ordinary P12 joint training is untouched."""

    original_set_trainable = P11DownstreamModel.set_backbone_trainable
    original_train = P11DownstreamModel.train
    original_groups = training._trainable_groups
    original_diagnostic = training.gradient_diagnostic

    def set_trainable(model: P11DownstreamModel, enabled: bool) -> None:
        set_phase_only_backbone_trainable(model, enabled)

    def train_mode(model: P11DownstreamModel, mode: bool = True):
        result = original_train(model, mode)
        if mode:
            # Frozen electronic mixers contain dropout.  Keep them in eval mode
            # while preserving autograd through the optical phase tensors.
            model.backbone.eval()
            model.head.train(True)
        return result

    P11DownstreamModel.set_backbone_trainable = set_trainable  # type: ignore[method-assign]
    P11DownstreamModel.train = train_mode  # type: ignore[method-assign]
    training._trainable_groups = phase_only_trainable_groups
    training.gradient_diagnostic = phase_only_gradient_diagnostic
    try:
        yield
    finally:
        P11DownstreamModel.set_backbone_trainable = original_set_trainable  # type: ignore[method-assign]
        P11DownstreamModel.train = original_train  # type: ignore[method-assign]
        training._trainable_groups = original_groups
        training.gradient_diagnostic = original_diagnostic


def run_phase_only(
    settings: PhaseOnlySettings, *, resume: bool = True
) -> dict[str, Any]:
    if not isinstance(settings, PhaseOnlySettings):
        raise TypeError("run_phase_only requires PhaseOnlySettings")
    if settings.method != "noft":
        noft_settings = load_phase_only_settings(
            settings.protocol.panel_config,
            task=settings.task,
            method="noft",
            seed=settings.seed,
            output_root=settings.paths.output_root,
        )
        noft_result_path = noft_settings.output_dir / "result.json"
        try:
            noft_result = json.loads(noft_result_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as error:
            raise RuntimeError(
                "Phase-only adaptation requires its own completed head-only result: "
                f"{noft_result_path}"
            ) from error
        expected_noft_digest = training.sha256_json(
            {
                "settings": noft_settings.to_dict(),
                "implementation_sha256": implementation_sha256(),
            }
        )
        expected_noft_panel = noft_settings.protocol.to_dict(method="noft")
        expected_noft = {
            "status": "complete",
            "task": settings.task,
            "method": "noft",
            "seed": settings.seed,
            "config_digest": expected_noft_digest,
            "implementation_sha256": implementation_sha256(),
            "phase_only_panel": expected_noft_panel,
        }
        mismatches = {
            key: (noft_result.get(key), expected)
            for key, expected in expected_noft.items()
            if noft_result.get(key) != expected
        }
        if mismatches:
            raise RuntimeError(
                "Phase-only head-only result identity mismatch: "
                f"{mismatches}"
            )
    with phase_only_runtime():
        result = training.run_training(settings, resume=resume)
    tag = settings.protocol.to_dict(method=settings.method)
    existing = result.get("phase_only_panel")
    if existing is not None and existing != tag:
        raise RuntimeError("Existing result has a different phase-only panel identity")
    result["phase_only_panel"] = tag
    training.write_json(settings.output_dir / "result.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one strict phase-only P12 transfer job."
    )
    add_phase_only_arguments(parser)
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", dest="resume", action="store_true")
    resume.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    return parser


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings: PhaseOnlySettings | None = None
    started = time.time()
    try:
        settings = load_phase_only_settings_from_args(args)
        result = run_phase_only(settings, resume=bool(args.resume))
        if result.get("status") != "complete":
            raise RuntimeError("phase-only training returned without a complete result")
        return 0
    except Exception as error:
        trace = traceback.format_exc()
        if settings is not None:
            _write_json_atomic(
                settings.output_dir / "failure.json",
                {
                    "status": "failed",
                    "task": settings.task,
                    "method": settings.method,
                    "seed": settings.seed,
                    "adaptation_scope": ADAPTATION_SCOPE,
                    "pid": os.getpid(),
                    "hostname": socket.gethostname(),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "started_at_unix": started,
                    "failed_at_unix": time.time(),
                    "exception_type": type(error).__name__,
                    "exception": str(error),
                    "traceback": trace,
                    "resume_requested": bool(args.resume),
                },
            )
        print(trace, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ADAPTATION_SCOPE",
    "PANEL_FORMAT",
    "PhaseOnlySettings",
    "add_phase_only_arguments",
    "load_phase_only_settings",
    "load_phase_only_settings_from_args",
    "panel_implementation_sha256",
    "parameter_family_report",
    "phase_only_gradient_diagnostic",
    "phase_only_runtime",
    "phase_only_trainable_groups",
    "run_phase_only",
    "set_phase_only_backbone_trainable",
]
