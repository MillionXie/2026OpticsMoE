"""Export deterministic probes and paired simulated CCD references.

This is the only agreement step that needs the Qwen/model environment.  The
resulting stage directory is self-contained and can be copied below the common
``experiments/.../hardware_sessions`` tree on the laboratory computer.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from experiments.hardware_sdk.workflows.reconstruct_slm import (
    encode_active_amplitude_with_metadata,
    reconstruct_directory,
    save_active_png,
)
from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust import (
    hardware_bridge as bridge,
)
from experiments.qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval.prepare_caltech101_retrieval import (
    prepare_caltech101_subset,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optical_artifacts import (
    phase_tensors,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.settings import (
    _nested,
    _read_config,
)

from .agreement_common import (
    SCHEMA_VERSION,
    STAGES,
    read_json,
    require_sha256,
    safe_relative_file,
    sha256_file,
    sha256_text,
    stage_directory,
    verify_file,
    write_csv,
    write_json,
)
from .modeling import build_hybrid_student, load_backbone
from .settings import load_settings


REFERENCE_KINDS = ("ideal_model_fp32", "transport_quantized")


def _existing_session_stages(
    session_dir: Path,
    *,
    checkpoint_sha256: str,
    resolved_config_sha256: str,
) -> list[dict[str, Any]]:
    """Load prior stages only when they belong to this exact export run."""

    root_path = session_dir / "agreement_manifest.json"
    if not root_path.is_file():
        return []
    root = read_json(root_path)
    if not isinstance(root, dict):
        raise RuntimeError("Existing agreement manifest must be a JSON object")
    if root.get("schema_version") != SCHEMA_VERSION or root.get("type") != (
        "qwen_warmstart5_sim_to_real_agreement_session"
    ):
        raise RuntimeError(
            "Existing agreement manifest has an incompatible schema/type; "
            "use a new session directory"
        )
    expected_checkpoint = require_sha256(
        root.get("checkpoint_sha256", ""), label="session checkpoint digest"
    )
    expected_config = require_sha256(
        root.get("resolved_config_sha256", ""), label="session resolved-config digest"
    )
    if expected_checkpoint != checkpoint_sha256 or expected_config != resolved_config_sha256:
        raise RuntimeError(
            "Existing agreement session was exported with a different checkpoint "
            "or resolved config; use a new session directory"
        )
    stages = root.get("stages")
    if not isinstance(stages, list):
        raise RuntimeError("Existing agreement manifest stages must be a list")
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for item in stages:
        if not isinstance(item, dict) or item.get("stage") not in STAGES:
            raise RuntimeError("Existing agreement manifest contains an invalid stage")
        name = str(item["stage"])
        if name in seen:
            raise RuntimeError(f"Existing agreement manifest repeats stage {name}")
        seen.add(name)
        expected_directory = stage_directory(session_dir, name).relative_to(
            session_dir
        ).as_posix()
        if item.get("directory") != expected_directory:
            raise RuntimeError(f"Existing agreement directory mismatch for {name}")
        contract_path = safe_relative_file(
            session_dir, item.get("contract", ""), label=f"{name} stage contract"
        )
        verify_file(
            contract_path,
            item.get("contract_sha256", ""),
            label=f"{name} stage contract",
        )
        contract = read_json(contract_path)
        if (
            not isinstance(contract, dict)
            or contract.get("stage") != name
            or contract.get("checkpoint_sha256") != checkpoint_sha256
            or contract.get("resolved_config_sha256") != resolved_config_sha256
        ):
            raise RuntimeError(f"Existing agreement contract provenance mismatch for {name}")
        result.append(dict(item))
    return result


def designed_probes(size: int = 478, seed: int = 42) -> dict[str, np.ndarray]:
    """Return a small, deterministic and spatially diverse uint8 probe suite."""

    if size < 64:
        raise ValueError("Agreement probes require at least a 64x64 active region")
    y, x = np.mgrid[:size, :size]
    center = (size - 1) / 2.0
    arm = max(6, int(round(size * 0.035)))
    margin = max(8, int(round(size * 0.08)))

    dark = np.zeros((size, size), dtype=np.uint8)
    uniform = np.full((size, size), 255, dtype=np.uint8)
    cross = np.zeros((size, size), dtype=np.uint8)
    cross[np.abs(x - center) <= arm] = 255
    cross[np.abs(y - center) <= arm] = 255
    cross[:margin] = 0
    cross[-margin:] = 0
    cross[:, :margin] = 0
    cross[:, -margin:] = 0

    quadrant = np.empty((size, size), dtype=np.uint8)
    half = size // 2
    quadrant[:half, :half] = 255
    quadrant[:half, half:] = 176
    quadrant[half:, :half] = 96
    quadrant[half:, half:] = 32

    period = 32.0
    grating_x = np.rint(127.5 * (1.0 + np.cos(2.0 * np.pi * x / period))).astype(np.uint8)
    grating_y = np.rint(127.5 * (1.0 + np.cos(2.0 * np.pi * y / period))).astype(np.uint8)
    checker = (((x // 24 + y // 24) % 2) * 255).astype(np.uint8)

    # A blockwise random probe covers many low/mid spatial frequencies without
    # producing an unphysical native-pixel white-noise target.
    generator = np.random.default_rng(int(seed))
    block = max(8, size // 24)
    coarse_h = int(np.ceil(size / block))
    coarse_w = int(np.ceil(size / block))
    coarse = generator.integers(0, 2, size=(coarse_h, coarse_w), dtype=np.uint8) * 255
    lowpass_random = np.repeat(np.repeat(coarse, block, axis=0), block, axis=1)[:size, :size]

    return {
        "dark": dark,
        "uniform": uniform,
        "center_cross": cross,
        "quadrant_code": quadrant,
        "grating_x_p32": grating_x,
        "grating_y_p32": grating_y,
        "checker_p48": checker,
        f"block_random_seed{int(seed)}": lowpass_random,
    }


def select_model_samples(samples: Iterable[Any], per_class: int) -> list[Any]:
    """Select held-out test samples deterministically, balanced by class."""

    if per_class <= 0:
        raise ValueError("per_class must be positive")
    grouped: dict[int, list[Any]] = defaultdict(list)
    for sample in samples:
        if str(sample.split) == "test":
            grouped[int(sample.sku_index)].append(sample)
    if not grouped:
        raise RuntimeError("No test samples are available for agreement export")
    selected: list[Any] = []
    for label in sorted(grouped):
        ordered = sorted(
            grouped[label],
            key=lambda sample: sha256_text(bridge._key(sample)),
        )
        if len(ordered) < per_class:
            raise RuntimeError(
                f"Class {label} has {len(ordered)} test samples, needs {per_class}"
            )
        selected.extend(ordered[:per_class])
    return selected


def _raw_phase_for_stage(replacement: Any, stage: str) -> torch.Tensor:
    branch = bridge._branch_for_stage(replacement, stage)
    values = phase_tensors(branch.core)
    key = "physical_expert_mosaic_rad" if stage.endswith("expert") else "physical_global_phase_rad"
    phase = values[key].detach()
    if phase.ndim != 2:
        raise RuntimeError(f"Agreement phase must be 2-D, got {tuple(phase.shape)}")
    return phase


def _decode_export_phase(encoded: np.ndarray, settings: Any) -> np.ndarray:
    """Undo export orientation and decode the exact uint8 phase command."""

    value = np.asarray(encoded, dtype=np.uint8)
    if bool(settings.hardware_phase_flip_horizontal):
        value = np.flip(value, axis=1)
    if bool(settings.hardware_phase_flip_vertical):
        value = np.flip(value, axis=0)
    return value.astype(np.float32) * (2.0 * np.pi / 256.0)


def simulate_active_field(
    branch: Any,
    amplitude: torch.Tensor,
    phase_rad: torch.Tensor,
) -> torch.Tensor:
    """Propagate a zero-phase active amplitude through one current-stage mask."""

    if amplitude.ndim == 2:
        amplitude = amplitude.unsqueeze(0)
    if phase_rad.ndim != 2 or amplitude.ndim != 3:
        raise ValueError("amplitude must be [B,H,W] and phase_rad must be [H,W]")
    active = branch.core.geometry.active_aperture
    expected = (active.y1 - active.y0, active.x1 - active.x0)
    if tuple(amplitude.shape[-2:]) != expected or tuple(phase_rad.shape) != expected:
        raise ValueError(
            f"Active agreement shapes must be {expected}, got "
            f"amplitude={tuple(amplitude.shape[-2:])}, phase={tuple(phase_rad.shape)}"
        )
    device = next(branch.parameters()).device
    amplitude = amplitude.to(device=device, dtype=torch.float32)
    phase_rad = phase_rad.to(device=device, dtype=torch.float32)
    canvas_size = int(branch.core.geometry.canvas_size)
    field = torch.zeros(
        amplitude.shape[0], canvas_size, canvas_size,
        device=device, dtype=torch.complex64,
    )
    field[:, active.y0 : active.y1, active.x0 : active.x1] = amplitude.to(torch.complex64)
    modulation = torch.ones_like(field)
    modulation[:, active.y0 : active.y1, active.x0 : active.x1] = torch.exp(
        1j * phase_rad
    )
    shifts = {"input": (0, 0), "phase": (0, 0), "ccd": (0, 0)}
    return branch._simulate_detector_roi(field, modulation, shifts).detach().cpu().float()


def _save_reference(path: Path, value: np.ndarray) -> str:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError("Theoretical CCD reference must be finite nonnegative 2-D intensity")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, intensity=array)
    return sha256_file(path)


def _agreement_config(config_path: Path) -> dict[str, Any]:
    raw = _read_config(config_path)
    value = dict(_nested(raw, "agreement", {}) or {})
    return {
        "task_test_per_class": int(value.get("task_test_per_class", 2)),
        "repeat_task_keys": int(value.get("repeat_task_keys", 2)),
        "repeats_per_key": int(value.get("repeats_per_key", 3)),
        "probe_seed": int(value.get("probe_seed", 42)),
        "batch_size": int(value.get("batch_size", 5)),
        "relative_clip": float(value.get("relative_clip", 12.0)),
        "log_compression": float(value.get("log_compression", 1.0)),
        "signal_energy_fraction": float(value.get("signal_energy_fraction", 0.99)),
        "bootstrap_samples": int(value.get("bootstrap_samples", 2000)),
    }


def _write_one_probe(
    *,
    stage_dir: Path,
    capture_key: str,
    canonical_key: str,
    amplitude_uint8: np.ndarray,
    amplitude_scale: float,
    ideal: np.ndarray,
    transport: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    compact_path = stage_dir / "compact_amplitude" / f"{capture_key}.png"
    save_active_png(np.asarray(amplitude_uint8, dtype=np.uint8), compact_path)
    ideal_path = stage_dir / "theoretical_ccd" / "ideal_model_fp32" / f"{capture_key}.npz"
    transport_path = stage_dir / "theoretical_ccd" / "transport_quantized" / f"{capture_key}.npz"
    ideal_sha = _save_reference(ideal_path, ideal)
    transport_sha = _save_reference(transport_path, transport)
    return {
        **metadata,
        "capture_key": capture_key,
        "canonical_key": canonical_key,
        "amplitude_file": f"{capture_key}.bmp",
        "compact_amplitude_file": compact_path.relative_to(stage_dir).as_posix(),
        "compact_amplitude_sha256": sha256_file(compact_path),
        "amplitude_encoding_scale": f"{float(amplitude_scale):.10g}",
        "ideal_reference_file": ideal_path.relative_to(stage_dir).as_posix(),
        "ideal_reference_sha256": ideal_sha,
        "transport_reference_file": transport_path.relative_to(stage_dir).as_posix(),
        "transport_reference_sha256": transport_sha,
    }


@torch.no_grad()
def export_stage(
    *,
    config_path: str | Path,
    checkpoint: str | Path,
    session_dir: str | Path,
    stage: str,
    upstream_source: str = "simulation",
) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    config_path = Path(config_path).expanduser().resolve()
    checkpoint = Path(checkpoint).expanduser().resolve()
    session_dir = Path(session_dir).expanduser().resolve()
    destination = stage_directory(session_dir, stage)
    if (destination / "probe_manifest.csv").exists():
        raise FileExistsError(
            f"Agreement stage already exists: {destination}. Use a new session directory."
        )
    agreement = _agreement_config(config_path)
    if agreement["repeat_task_keys"] < 0 or agreement["repeats_per_key"] < 1:
        raise ValueError("Invalid agreement repeat configuration")
    if agreement["repeat_task_keys"] and agreement["repeats_per_key"] < 2:
        raise ValueError("Repeated keys require repeats_per_key >= 2")
    if agreement["batch_size"] <= 0:
        raise ValueError("agreement.batch_size must be positive")

    resolved_raw = _read_config(config_path)
    resolved_config_sha = sha256_text(
        json.dumps(resolved_raw, ensure_ascii=False, sort_keys=True, default=str)
    )
    checkpoint_sha = sha256_file(checkpoint)
    # Validate the session before loading Qwen or writing any stage payload.
    existing_stages = _existing_session_stages(
        session_dir,
        checkpoint_sha256=checkpoint_sha,
        resolved_config_sha256=resolved_config_sha,
    )

    settings = load_settings(config_path)
    if bool(settings.hardware_ccd_flip_vertical) or bool(
        settings.hardware_ccd_flip_horizontal
    ):
        raise ValueError(
            "Agreement export requires canonical model-coordinate CCD settings "
            "with hardware.ccd.flip_vertical=false and flip_horizontal=false. "
            "The detector homography already resolves orientation; a downstream "
            "flip would apply it twice."
        )
    bridge.build_hybrid_student = build_hybrid_student
    bridge.load_backbone = load_backbone
    bridge.load_settings = load_settings
    bundle = prepare_caltech101_subset(settings, persist=True)
    selected = select_model_samples(bundle.all_samples(), agreement["task_test_per_class"])
    loaded, replacement, readout = bridge._load_model(settings, checkpoint)
    branch = bridge._branch_for_stage(replacement, stage)
    measured_upstream = bridge._measurement_plan(stage, upstream_source, include_current=False)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "ccd_captured").mkdir(exist_ok=True)
    (destination / "acquisition_logs").mkdir(exist_ok=True)

    raw_phase = _raw_phase_for_stage(replacement, stage)
    export_phase = bridge._phase_for_stage(replacement, stage, settings)
    compact_phase = destination / "compact_phase" / f"{stage}.png"
    save_active_png(export_phase, compact_phase)
    reconstruct_directory(
        compact_phase.parent,
        destination / "phase_to_play",
        slm_size_wh=(settings.hardware_phase_slm_width, settings.hardware_phase_slm_height),
        scale_factor=None,
        center_xy=(settings.hardware_phase_slm_center_x, settings.hardware_phase_slm_center_y),
        logical_pixel_pitch_um=settings.language_optical_pixel_pitch_um,
        slm_pixel_pitch_um=settings.hardware_phase_slm_pixel_pitch_um,
    )
    decoded_phase = torch.from_numpy(_decode_export_phase(export_phase, settings).copy())

    rows: list[dict[str, Any]] = []
    common = {
        "stage": stage,
        "upstream_source": upstream_source,
    }
    try:
        # Human-designed probes are a calibration split and never contribute to
        # task/test agreement summaries.
        probes = designed_probes(int(settings.hardware_ccd_target_size), agreement["probe_seed"])
        for name, encoded in probes.items():
            amplitude = torch.from_numpy(encoded.astype(np.float32) / 255.0)
            ideal = simulate_active_field(branch, amplitude, raw_phase)[0].numpy()
            transport = simulate_active_field(branch, amplitude, decoded_phase)[0].numpy()
            rows.append(
                _write_one_probe(
                    stage_dir=destination,
                    capture_key=f"probe__{name}",
                    canonical_key=f"probe__{name}",
                    amplitude_uint8=encoded,
                    amplitude_scale=1.0,
                    ideal=ideal,
                    transport=transport,
                    metadata={
                        **common,
                        "source_kind": "designed",
                        "role": "calibration",
                        "split": "calibration",
                        "sku_index": "",
                        "sku_name": "",
                        "sample_id": "",
                        "repeat_index": 0,
                    },
                )
            )

        originals: dict[str, tuple[dict[str, Any], np.ndarray, float, np.ndarray, np.ndarray]] = {}
        batch_size = agreement["batch_size"]
        for start in range(0, len(selected), batch_size):
            batch = selected[start : start + batch_size]
            keys = [bridge._key(sample) for sample in batch]
            bridge._install_measurements(
                replacement,
                settings,
                session_dir,
                keys,
                measured_stages=measured_upstream,
            )
            bridge._forward_samples(loaded, replacement, readout, settings, batch)
            amplitude_batch = bridge._amplitude_for_stage(branch, stage).detach().cpu().float()
            ideal_batch = (
                branch.last_raw_expert_ccd if stage.endswith("expert") else branch.last_raw_ccd
            )
            if ideal_batch is None or len(ideal_batch) != len(batch):
                raise RuntimeError(f"No deterministic simulated CCD was captured for {stage}")
            ideal_batch = ideal_batch.detach().cpu().float()
            for sample, key, amplitude, ideal in zip(batch, keys, amplitude_batch, ideal_batch):
                encoded, encoding = encode_active_amplitude_with_metadata(amplitude.numpy())
                # The laboratory SLM receives only the normalized uint8 frame.
                # ``encoding["scale"]`` is provenance for reconstructing the
                # pre-export tensor; it is not a gain command understood by the
                # hardware and therefore must not be multiplied back into the
                # transport-matched optical simulation.
                decoded_amplitude = torch.from_numpy(
                    encoded.astype(np.float32) / 255.0
                )
                transport = simulate_active_field(branch, decoded_amplitude, decoded_phase)[0].numpy()
                metadata = {
                    **common,
                    "source_kind": "model",
                    "role": "evaluation",
                    "split": str(sample.split),
                    "sku_index": int(sample.sku_index),
                    "sku_name": str(sample.sku_name),
                    "sample_id": str(sample.sample_id),
                    "repeat_index": 0,
                }
                row = _write_one_probe(
                    stage_dir=destination,
                    capture_key=key,
                    canonical_key=key,
                    amplitude_uint8=encoded,
                    amplitude_scale=float(encoding["scale"]),
                    ideal=ideal.numpy(),
                    transport=transport,
                    metadata=metadata,
                )
                rows.append(row)
                originals[key] = (metadata, encoded, float(encoding["scale"]), ideal.numpy(), transport)
            print(f"[agreement_export:{stage}] model_inputs={min(start + len(batch), len(selected))}/{len(selected)}", flush=True)

        repeat_keys = [bridge._key(sample) for sample in selected[: agreement["repeat_task_keys"]]]
        for key in repeat_keys:
            metadata, encoded, scale, ideal, transport = originals[key]
            for repeat_index in range(1, agreement["repeats_per_key"]):
                rows.append(
                    _write_one_probe(
                        stage_dir=destination,
                        capture_key=f"{key}__r{repeat_index:02d}",
                        canonical_key=key,
                        amplitude_uint8=encoded,
                        amplitude_scale=scale,
                        ideal=ideal,
                        transport=transport,
                        metadata={**metadata, "repeat_index": repeat_index},
                    )
                )
    finally:
        bridge._clear_measurements(replacement)
        replacement.close()

    rows = [{"order": index, **row} for index, row in enumerate(rows)]
    fields = list(rows[0])
    manifest_path = destination / "probe_manifest.csv"
    write_csv(manifest_path, rows, fields)
    compact_rows = [
        {
            "order": row["order"],
            "capture_key": row["capture_key"],
            "filename": Path(row["compact_amplitude_file"]).name,
            "sha256": row["compact_amplitude_sha256"],
            "encoding_scale": row["amplitude_encoding_scale"],
        }
        for row in rows
    ]
    write_csv(
        destination / "compact_amplitude_manifest.csv",
        compact_rows,
        list(compact_rows[0]),
    )

    stage_contract = {
        "schema_version": SCHEMA_VERSION,
        "type": "qwen_warmstart5_sim_to_real_agreement",
        "stage": stage,
        "upstream_source": upstream_source,
        "measured_upstream_stages": list(measured_upstream),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "source_config": str(config_path),
        "source_config_sha256": sha256_file(config_path),
        "resolved_config_sha256": resolved_config_sha,
        "probe_manifest": manifest_path.name,
        "probe_manifest_sha256": sha256_file(manifest_path),
        "probe_count": len(rows),
        "calibration_probe_count": sum(row["role"] == "calibration" for row in rows),
        "evaluation_capture_count": sum(row["role"] == "evaluation" for row in rows),
        "independent_evaluation_probe_count": len(
            {row["canonical_key"] for row in rows if row["role"] == "evaluation"}
        ),
        "reference_kinds": list(REFERENCE_KINDS),
        "active_shape_hw": [int(settings.hardware_ccd_target_size)] * 2,
        "compact_phase_file": compact_phase.relative_to(destination).as_posix(),
        "compact_phase_sha256": sha256_file(compact_phase),
        "phase_reconstruction_manifest": "phase_to_play/reconstruction_manifest.csv",
        "phase_reconstruction_manifest_sha256": sha256_file(
            destination / "phase_to_play" / "reconstruction_manifest.csv"
        ),
        "phase_transport_encoding": {
            "encoder": "floor(mod(phase_rad,2pi)*256/(2pi)) clipped to uint8",
            "decoder": "gray_uint8*2pi/256",
            "bin_count": 256,
            "endpoint_convention": "2pi is exclusive; gray 255 represents 255/256 turn",
            "source_function": "experiments.hardware_sdk.workflows.reconstruct_slm.encode_active_phase",
        },
        "agreement": agreement,
        "coordinate_contract": (
            "Theoretical CCD is in model coordinates. Captured CCD must be mapped "
            "once by the session-level four-corner transform before evaluation."
        ),
        "canonical_ccd_contract": {
            "saved_frame_orientation": "canonical_model_xy",
            "orientation_canonicalized": True,
            "downstream_loader_flip_required": False,
            "configured_flip_vertical": False,
            "configured_flip_horizontal": False,
        },
        "preprocessing_contract": (
            "No background subtraction, per-frame min-max stretch, histogram "
            "matching, gamma fit, or per-frame registration is permitted."
        ),
    }
    contract_path = destination / "agreement_contract.json"
    write_json(contract_path, stage_contract)

    root_manifest_path = session_dir / "agreement_manifest.json"
    stages = [item for item in existing_stages if item.get("stage") != stage]
    stages.append(
        {
            "stage": stage,
            "directory": destination.relative_to(session_dir).as_posix(),
            "contract": f"{destination.relative_to(session_dir).as_posix()}/agreement_contract.json",
            "contract_sha256": sha256_file(contract_path),
        }
    )
    stages.sort(key=lambda item: STAGES.index(item["stage"]))
    write_json(
        root_manifest_path,
        {
            "schema_version": SCHEMA_VERSION,
            "type": "qwen_warmstart5_sim_to_real_agreement_session",
            "checkpoint_sha256": checkpoint_sha,
            "resolved_config_sha256": resolved_config_sha,
            "stages": stages,
        },
    )
    print(
        f"[agreement_export] stage={stage} probes={len(rows)} "
        f"session={session_dir}",
        flush=True,
    )
    return stage_contract


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export deterministic Qwen warmstart5 sim-to-real CCD probes"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=["language_global"])
    parser.add_argument(
        "--upstream-source",
        choices=("simulation", "measured"),
        default="simulation",
        help="simulation isolates each current plane; measured enables sequential four-stage export",
    )
    args = parser.parse_args()
    for stage in args.stages:
        export_stage(
            config_path=args.config,
            checkpoint=args.checkpoint,
            session_dir=args.session_dir,
            stage=stage,
            upstream_source=args.upstream_source,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
