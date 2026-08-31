from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from .migration import migrate_strict_p11_checkpoint, sha256_file
from .model import (
    P13_SUPPORTED_DEPTHS,
    QwenStemProgressiveOpticalImageNetBackbone,
)


SWEEP_FORMAT = "p13-gpu-engineering-sweep-v1"
RESULT_FORMAT = "p13-gpu-engineering-depth-result-v1"
CLAIM_SCOPE = "engineering_only_not_accuracy_or_backbone_performance"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_depths(value: str) -> tuple[int, ...]:
    """Parse one depth or an ordered comma-separated depth sweep."""

    fields = [field.strip() for field in value.split(",")]
    if not fields or any(not field for field in fields):
        raise argparse.ArgumentTypeError(
            "depths must be one integer or a comma-separated integer list"
        )
    try:
        parsed = [int(field) for field in fields]
    except ValueError as error:
        raise argparse.ArgumentTypeError("depths must contain only integers") from error
    unsupported = [depth for depth in parsed if depth not in P13_SUPPORTED_DEPTHS]
    if unsupported:
        raise argparse.ArgumentTypeError(
            f"unsupported depths {unsupported}; choose from {P13_SUPPORTED_DEPTHS}"
        )
    # Do not spend a second full GPU run on a repeated CSV entry.
    return tuple(dict.fromkeys(parsed))


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write JSON through an adjacent temporary file and atomic replace."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _finite_number(value: float) -> bool:
    return math.isfinite(float(value))


def summarize_phase_gradients(
    named_parameters: Iterable[tuple[str, torch.Tensor]],
) -> dict[str, Any]:
    """Summarize the required present/finite/nonzero checks on CPU or CUDA."""

    total = 0
    present = 0
    finite = 0
    nonzero = 0
    norms: list[float] = []
    missing_names: list[str] = []
    nonfinite_names: list[str] = []
    zero_names: list[str] = []
    present_entries: list[tuple[str, torch.Tensor]] = []
    for name, parameter in named_parameters:
        total += 1
        gradient = parameter.grad
        if gradient is None:
            missing_names.append(name)
            continue
        present += 1
        present_entries.append((name, gradient))
    # Queue one reduction kernel per phase but perform only one device-to-host
    # transfer for each reduction family. A per-phase .item() would inject up
    # to 92 CUDA synchronizations and invalidate the throughput measurement.
    finite_values = (
        torch.stack([torch.isfinite(gradient).all() for _, gradient in present_entries])
        .detach()
        .cpu()
        .tolist()
        if present_entries
        else []
    )
    norm_values = (
        torch.stack(
            [gradient.detach().float().norm() for _, gradient in present_entries]
        )
        .cpu()
        .tolist()
        if present_entries
        else []
    )
    for (name, _), is_finite, norm_value in zip(
        present_entries, finite_values, norm_values
    ):
        if not is_finite:
            nonfinite_names.append(name)
            continue
        finite += 1
        norm = float(norm_value)
        norms.append(norm)
        if norm > 0.0:
            nonzero += 1
        else:
            zero_names.append(name)
    return {
        "new_phase_count": total,
        "gradient_present_count": present,
        "gradient_finite_count": finite,
        "gradient_nonzero_count": nonzero,
        "every_gradient_present": present == total,
        "every_gradient_finite": finite == total,
        "every_gradient_nonzero": nonzero == total,
        "minimum_gradient_norm": min(norms) if norms else None,
        "maximum_gradient_norm": max(norms) if norms else None,
        "missing_gradient_names": missing_names,
        "nonfinite_gradient_names": nonfinite_names,
        "zero_gradient_names": zero_names,
    }


def validate_result_fields(payload: Mapping[str, Any]) -> None:
    """Validate the stable, machine-readable contract used by sweep results."""

    required = {
        "format",
        "status",
        "claim_scope",
        "formal_training_started",
        "depth",
        "configuration",
        "source",
        "device",
        "alpha",
        "migration",
        "parameters",
        "checks",
        "measurement",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"engineering result is missing fields: {missing}")
    if payload["format"] != RESULT_FORMAT:
        raise ValueError("unexpected engineering result format")
    if payload["claim_scope"] != CLAIM_SCOPE:
        raise ValueError("engineering result must carry the non-performance scope")
    if payload["formal_training_started"] is not False:
        raise ValueError("engineering sweep must not be marked as formal training")
    source = payload["source"]
    device = payload["device"]
    if not isinstance(source, Mapping) or not source.get("p11_checkpoint_sha256"):
        raise ValueError("engineering result must report the P11 source SHA-256")
    if not source.get("stem_checkpoint_sha256"):
        raise ValueError("engineering result must report the stem SHA-256")
    if not isinstance(device, Mapping) or not device.get("gpu_uuid"):
        raise ValueError("engineering result must report the GPU UUID")
    if payload["status"] == "passed_engineering":
        passed_requirements = {
            "alpha.configured_epsilon": payload["alpha"].get("configured_epsilon"),
            "migration.source_checkpoint_sha256": payload["migration"].get(
                "source_checkpoint_sha256"
            ),
            "parameters.optical_phase_parameters": payload["parameters"].get(
                "optical_phase_parameters"
            ),
            "measurement.peak_allocated_bytes": payload["measurement"].get(
                "peak_allocated_bytes"
            ),
            "measurement.peak_reserved_bytes": payload["measurement"].get(
                "peak_reserved_bytes"
            ),
            "measurement.samples_per_second": payload["measurement"].get(
                "samples_per_second"
            ),
        }
        absent = sorted(
            name for name, value in passed_requirements.items() if value is None
        )
        if absent:
            raise ValueError(f"passed engineering result is missing fields: {absent}")


def _gpu_uuid(device: torch.device) -> str:
    properties = torch.cuda.get_device_properties(device)
    direct = getattr(properties, "uuid", None)
    if direct:
        return str(direct)
    getter = getattr(torch.cuda, "get_device_uuid", None)
    if callable(getter):
        direct = getter(device)
        if direct:
            return str(direct)

    # Resolve the physical index when CUDA_VISIBLE_DEVICES remaps cuda:0.
    logical_index = (
        torch.cuda.current_device() if device.index is None else device.index
    )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible_tokens = [token.strip() for token in visible.split(",") if token.strip()]
    physical_token = (
        visible_tokens[logical_index]
        if logical_index < len(visible_tokens)
        else str(logical_index)
    )
    if physical_token.startswith("GPU-"):
        return physical_token
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
        mapping = {
            fields[0].strip(): fields[1].strip()
            for row in rows
            if len(fields := row.split(",", maxsplit=1)) == 2
        }
        if physical_token in mapping:
            return mapping[physical_token]
    except (OSError, subprocess.SubprocessError):
        pass
    return f"unresolved-for-{device}"


def _device_report(device: torch.device) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "torch_device": str(device),
        "logical_cuda_index": torch.cuda.current_device()
        if device.index is None
        else device.index,
        "gpu_name": properties.name,
        "gpu_uuid": _gpu_uuid(device),
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": int(properties.total_memory),
        "free_memory_bytes_before_model": int(free_bytes),
        "mem_get_info_total_bytes": int(total_bytes),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _synthetic_field(
    *, batch_size: int, device: torch.device, seed: int
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    field = torch.rand(
        batch_size,
        3,
        224,
        224,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    rms = field.square().mean(dim=(-2, -1), keepdim=True).add(1.0e-5).sqrt()
    return (field / rms).detach()


def _weighted_field_loss(output: torch.Tensor) -> torch.Tensor:
    # A spatially and spectrally non-uniform probe avoids the nearly constant
    # energy objective that would make an ideal phase-only transform uninformative.
    spatial = torch.linspace(
        0.2,
        1.0,
        output.shape[-2],
        device=output.device,
        dtype=output.dtype,
    ).view(1, 1, -1, 1)
    spectral = torch.linspace(
        0.75,
        1.25,
        output.shape[1],
        device=output.device,
        dtype=output.dtype,
    ).view(1, -1, 1, 1)
    return (output.square() * spatial * spectral).mean()


def _optimizer(
    model: QwenStemProgressiveOpticalImageNetBackbone,
    *,
    phase_learning_rate: float,
    electronic_learning_rate: float,
) -> tuple[torch.optim.Optimizer, dict[str, int]]:
    phase_parameters = [parameter for parameter in model.phase_parameters()]
    phase_ids = {id(parameter) for parameter in phase_parameters}
    body_electronic = [
        parameter
        for slot in model.slots
        for parameter in slot.stage.parameters()
        if parameter.requires_grad and id(parameter) not in phase_ids
    ]
    optimizer = torch.optim.SGD(
        [
            {"params": phase_parameters, "lr": phase_learning_rate},
            {"params": body_electronic, "lr": electronic_learning_rate},
        ],
        momentum=0.0,
        weight_decay=0.0,
    )
    return optimizer, {
        "phase_parameter_count": sum(p.numel() for p in phase_parameters),
        "body_electronic_parameter_count": sum(p.numel() for p in body_electronic),
        "optimizer_parameter_count": sum(p.numel() for p in phase_parameters)
        + sum(p.numel() for p in body_electronic),
    }


def _one_step(
    *,
    model: QwenStemProgressiveOpticalImageNetBackbone,
    optimizer: torch.optim.Optimizer,
    field_template: torch.Tensor,
    audit_new_phase_gradients: bool,
) -> tuple[float, dict[str, Any] | None, bool]:
    optimizer.zero_grad(set_to_none=True)
    # P13 enables checkpointing only when the optical field requires a gradient.
    field = field_template.detach().requires_grad_(True)
    output, _ = model.forward_field(field)
    loss = _weighted_field_loss(output)
    loss_value = float(loss.detach().item())
    if not _finite_number(loss_value):
        raise FloatingPointError(
            f"non-finite synthetic optical-field loss: {loss_value}"
        )
    loss.backward()
    gradient_report: dict[str, Any] | None = None
    optimizer_step_performed = True
    if audit_new_phase_gradients:
        new_phase_parameters = [
            (f"slots.{slot.stage_index}.stage.raw_phase", slot.stage.raw_phase)
            for slot in model.new_slots()
        ]
        gradient_report = summarize_phase_gradients(new_phase_parameters)
        if not gradient_report["every_gradient_finite"]:
            # Do not write non-finite values into the migrated source state.
            optimizer_step_performed = False
    if optimizer_step_performed:
        optimizer.step()
    return loss_value, gradient_report, optimizer_step_performed


def _empty_result(
    *,
    depth: int,
    args: argparse.Namespace,
    source_sha256: str,
    stem_sha256: str,
    device_report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format": RESULT_FORMAT,
        "status": "running",
        "claim_scope": CLAIM_SCOPE,
        "formal_training_started": False,
        "created_at_utc": utc_now(),
        "completed_at_utc": None,
        "depth": depth,
        "configuration": {
            "batch_size": args.batch_size,
            "warmup_steps": args.warmup_steps,
            "measurement_steps": args.measurement_steps,
            "alpha_epsilon": args.alpha_epsilon,
            "activation_checkpointing": args.activation_checkpointing,
            "phase_learning_rate": args.phase_learning_rate,
            "electronic_learning_rate": args.electronic_learning_rate,
            "seed": args.seed,
            "synthetic_input": "post-adapter_optical_field_3x224x224_float32",
            "optimizer": "SGD(momentum=0,weight_decay=0)",
            "optimizer_scope": "optical_body_phases_and_stage_electronics_only",
        },
        "source": {
            "p11_checkpoint": str(args.p11_checkpoint.expanduser().resolve()),
            "p11_checkpoint_sha256": source_sha256,
            "stem_checkpoint": str(args.stem_checkpoint.expanduser().resolve()),
            "stem_checkpoint_sha256": stem_sha256,
        },
        "device": dict(device_report),
        "alpha": {},
        "migration": {},
        "parameters": {},
        "checks": {},
        "measurement": {},
    }


def run_depth(
    *,
    depth: int,
    args: argparse.Namespace,
    device: torch.device,
    device_report: Mapping[str, Any],
    source_sha256: str,
    stem_sha256: str,
) -> dict[str, Any]:
    result = _empty_result(
        depth=depth,
        args=args,
        source_sha256=source_sha256,
        stem_sha256=stem_sha256,
        device_report=device_report,
    )
    model: QwenStemProgressiveOpticalImageNetBackbone | None = None
    optimizer: torch.optim.Optimizer | None = None
    field: torch.Tensor | None = None
    try:
        torch.cuda.reset_peak_memory_stats(device)
        torch.manual_seed(args.seed + depth)
        torch.cuda.manual_seed_all(args.seed + depth)
        model = QwenStemProgressiveOpticalImageNetBackbone(
            args.stem_checkpoint,
            {
                "num_stages": depth,
                "seed": args.seed,
                "new_stage_alpha_init": args.alpha_epsilon,
                "new_stage_alpha_epsilon": args.alpha_epsilon,
                "new_stage_ramp_epochs": 10,
                "activation_checkpointing": args.activation_checkpointing,
            },
        )
        migration = migrate_strict_p11_checkpoint(model, args.p11_checkpoint)
        if migration["source_checkpoint_sha256"] != source_sha256:
            raise RuntimeError(
                "P11 source changed while the engineering sweep was running"
            )
        model.set_new_stage_alpha(args.alpha_epsilon)
        model.train()
        result["migration"] = migration
        result["alpha"] = {
            "configured_epsilon": args.alpha_epsilon,
            "report": model.depth_alpha_report(),
            "anchor_alpha": 1.0,
            "new_phase_gradient_gate_is_positive": args.alpha_epsilon > 0.0,
        }
        model_report = model.parameter_report()
        # Avoid duplicating the full migration manifest in two places.
        model_report["migration_manifest"] = "see top-level migration"
        result["parameters"] = model_report
        model.to(device)

        optimizer, optimizer_counts = _optimizer(
            model,
            phase_learning_rate=args.phase_learning_rate,
            electronic_learning_rate=args.electronic_learning_rate,
        )
        result["parameters"].update(optimizer_counts)
        field = _synthetic_field(
            batch_size=args.batch_size,
            device=device,
            seed=args.seed + 100_003 * depth,
        )
        torch.cuda.synchronize(device)
        model_input_allocated = torch.cuda.memory_allocated(device)
        model_input_reserved = torch.cuda.memory_reserved(device)

        warmup_losses: list[float] = []
        for _ in range(args.warmup_steps):
            loss, _, _ = _one_step(
                model=model,
                optimizer=optimizer,
                field_template=field,
                audit_new_phase_gradients=False,
            )
            warmup_losses.append(loss)

        # Run one dedicated untimed gradient audit. Keeping phase reductions
        # outside the timed loop makes samples/s describe an optimizer step,
        # not the diagnostics overhead.
        audit_loss, gradient_audit, audit_optimizer_step = _one_step(
            model=model,
            optimizer=optimizer,
            field_template=field,
            audit_new_phase_gradients=True,
        )
        if gradient_audit is None:
            raise RuntimeError("internal error: the dedicated phase audit was omitted")
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        measurement_start_allocated = torch.cuda.memory_allocated(device)
        measurement_start_reserved = torch.cuda.memory_reserved(device)

        measured_losses: list[float] = []
        step_seconds: list[float] = []
        for _ in range(args.measurement_steps):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            loss, _, _ = _one_step(
                model=model,
                optimizer=optimizer,
                field_template=field,
                audit_new_phase_gradients=False,
            )
            torch.cuda.synchronize(device)
            step_seconds.append(time.perf_counter() - started)
            measured_losses.append(loss)

        all_losses = warmup_losses + [audit_loss] + measured_losses
        every_present = gradient_audit["every_gradient_present"]
        every_finite = gradient_audit["every_gradient_finite"]
        every_nonzero = gradient_audit["every_gradient_nonzero"]
        losses_finite = all(_finite_number(value) for value in all_losses)
        result["checks"] = {
            "loss_finite_every_step": losses_finite,
            "every_new_phase_gradient_present": every_present,
            "every_new_phase_gradient_finite": every_finite,
            "every_new_phase_gradient_nonzero": every_nonzero,
            "gradient_audit_optimizer_step_performed": audit_optimizer_step,
            "gradient_audit_report": gradient_audit,
        }
        total_seconds = sum(step_seconds)
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        result["measurement"] = {
            "warmup_losses": warmup_losses,
            "gradient_audit_loss": audit_loss,
            "measurement_losses": measured_losses,
            "step_seconds": step_seconds,
            "mean_step_seconds": total_seconds / len(step_seconds),
            "minimum_step_seconds": min(step_seconds),
            "maximum_step_seconds": max(step_seconds),
            "samples_per_second": args.batch_size * len(step_seconds) / total_seconds,
            "model_and_input_allocated_bytes_before_steps": int(model_input_allocated),
            "model_and_input_reserved_bytes_before_steps": int(model_input_reserved),
            "measurement_start_allocated_bytes": int(measurement_start_allocated),
            "measurement_start_reserved_bytes": int(measurement_start_reserved),
            "peak_allocated_bytes": int(peak_allocated),
            "peak_reserved_bytes": int(peak_reserved),
            "incremental_peak_allocated_bytes": int(
                max(peak_allocated - measurement_start_allocated, 0)
            ),
            "incremental_peak_reserved_bytes": int(
                max(peak_reserved - measurement_start_reserved, 0)
            ),
            "timing_excludes_dedicated_gradient_audit": True,
        }
        passed = losses_finite and every_present and every_finite and every_nonzero
        result["status"] = "passed_engineering" if passed else "failed_checks"
    except torch.cuda.OutOfMemoryError as error:
        result["status"] = "failed_oom"
        result["error"] = {"type": type(error).__name__, "message": str(error)}
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            result["status"] = "failed_oom"
        else:
            result["status"] = "failed_error"
        result["error"] = {"type": type(error).__name__, "message": str(error)}
    except Exception as error:  # Keep later depths runnable and preserve diagnostics.
        result["status"] = "failed_error"
        result["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        if result["status"] == "failed_oom":
            result["measurement"].update(
                {
                    "peak_allocated_bytes_before_oom": int(
                        torch.cuda.max_memory_allocated(device)
                    ),
                    "peak_reserved_bytes_before_oom": int(
                        torch.cuda.max_memory_reserved(device)
                    ),
                }
            )
        result["completed_at_utc"] = utc_now()
        del optimizer
        del model
        del field
        gc.collect()
        torch.cuda.empty_cache()
        try:
            torch.cuda.synchronize(device)
        except RuntimeError:
            # Preserve the original per-depth failure and let the caller try
            # the next depth after cleanup.
            pass
    validate_result_fields(result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Engineering-only P13 CUDA memory/throughput/gradient sweep; this is "
            "not ImageNet training and produces no performance claim"
        )
    )
    parser.add_argument("--stem-checkpoint", type=Path, required=True)
    parser.add_argument("--p11-checkpoint", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--depths", type=parse_depths, default=parse_depths("16,32,64,100")
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measurement-steps", type=int, default=2)
    parser.add_argument("--alpha-epsilon", type=float, default=0.01)
    parser.add_argument("--phase-learning-rate", type=float, default=1.0e-2)
    parser.add_argument("--electronic-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if args.measurement_steps <= 0:
        raise ValueError("measurement_steps must be positive")
    if not 0.0 < args.alpha_epsilon <= 1.0:
        raise ValueError("alpha_epsilon must lie in (0,1]")
    if args.phase_learning_rate <= 0.0 or args.electronic_learning_rate <= 0.0:
        raise ValueError("learning rates must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("P13 engineering sweep requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.cuda.set_device(device)
    source_sha256 = sha256_file(args.p11_checkpoint)
    stem_sha256 = sha256_file(args.stem_checkpoint)
    device_report = _device_report(device)
    output = args.output_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "format": SWEEP_FORMAT,
        "claim_scope": CLAIM_SCOPE,
        "formal_training_started": False,
        "created_at_utc": utc_now(),
        "completed_at_utc": None,
        "requested_depths": list(args.depths),
        "source_p11_checkpoint_sha256": source_sha256,
        "source_stem_checkpoint_sha256": stem_sha256,
        "gpu_uuid": device_report["gpu_uuid"],
        "results": [],
    }
    atomic_write_json(output / "sweep_summary.json", summary)
    for depth in args.depths:
        result = run_depth(
            depth=depth,
            args=args,
            device=device,
            device_report=device_report,
            source_sha256=source_sha256,
            stem_sha256=stem_sha256,
        )
        result_path = output / f"depth_{depth:03d}" / "result.json"
        atomic_write_json(result_path, result)
        summary["results"].append(
            {
                "depth": depth,
                "status": result["status"],
                "result_json": str(result_path),
            }
        )
        summary["completed_at_utc"] = utc_now()
        atomic_write_json(output / "sweep_summary.json", summary)
        print(
            json.dumps(
                {
                    "depth": depth,
                    "status": result["status"],
                    "result_json": str(result_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    passed = all(row["status"] == "passed_engineering" for row in summary["results"])
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
