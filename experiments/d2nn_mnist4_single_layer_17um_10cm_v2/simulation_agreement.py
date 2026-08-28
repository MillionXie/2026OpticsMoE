"""Compare measured MNIST-4 CCD frames with the exact played-BMP simulation.

The classification path remains raw: no CCD normalization, nonlinearity,
background subtraction, or resize is introduced here.  Scale-insensitive shape
normalizations are used only to report sim-to-real agreement metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.agreement_evaluate import (
    pair_metrics,
)

from .lab_session import ACTIVE_BOUNDS, PHASE_SIZE, _read_json, validate_session
from .settings import load_settings


DETECTOR_BOUNDS = (
    (162, 162, 221, 221),
    (257, 162, 316, 221),
    (162, 257, 221, 316),
    (257, 257, 316, 316),
)
CM_TO_INCH = 1.0 / 2.54


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("Cannot write empty agreement results")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _native_source_indexes(
    source_size: int,
    output_size: int,
    *,
    logical_pitch_um: float,
    native_pitch_um: float,
) -> np.ndarray:
    physical = (
        np.arange(output_size, dtype=np.float64) + 0.5 - output_size / 2.0
    ) * native_pitch_um
    indexes = np.floor(physical / logical_pitch_um + source_size / 2.0)
    return np.clip(indexes.astype(np.int64), 0, source_size - 1)


def inverse_physical_pitch_nearest(
    native: np.ndarray,
    *,
    logical_size: int = 478,
    logical_pitch_um: float = 17.0,
    native_pitch_um: float = 8.0,
) -> np.ndarray:
    """Recover one representative native pixel for every logical phase pixel."""

    value = np.asarray(native)
    if value.ndim != 2 or value.dtype != np.uint8:
        raise ValueError("native phase crop must be a 2-D uint8 array")
    y_map = _native_source_indexes(
        logical_size,
        value.shape[0],
        logical_pitch_um=logical_pitch_um,
        native_pitch_um=native_pitch_um,
    )
    x_map = _native_source_indexes(
        logical_size,
        value.shape[1],
        logical_pitch_um=logical_pitch_um,
        native_pitch_um=native_pitch_um,
    )

    def representatives(mapping: np.ndarray) -> np.ndarray:
        selected: list[int] = []
        for logical_index in range(logical_size):
            positions = np.flatnonzero(mapping == logical_index)
            if positions.size == 0:
                raise RuntimeError(
                    f"Native raster contains no sample for logical index {logical_index}"
                )
            selected.append(int(positions[len(positions) // 2]))
        return np.asarray(selected, dtype=np.int64)

    return value[np.ix_(representatives(y_map), representatives(x_map))]


def decode_played_phase(phase_bmp: Path, contract: dict[str, Any]) -> np.ndarray:
    phase_export = contract.get("phase_export")
    if not isinstance(phase_export, dict):
        raise RuntimeError("stage_contract.json has no phase_export contract")
    with Image.open(phase_bmp) as opened:
        opened.load()
        if opened.mode != "L" or opened.size != PHASE_SIZE:
            raise RuntimeError(
                f"Phase BMP must be L/{PHASE_SIZE}, got {opened.mode}/{opened.size}"
            )
        full = np.asarray(opened, dtype=np.uint8)
    left, top, right, bottom = map(
        int, phase_export["native_active_bounds_xyxy"]
    )
    native = np.ascontiguousarray(full[top:bottom, left:right])
    logical = inverse_physical_pitch_nearest(
        native,
        logical_size=int(phase_export["logical_shape_hw"][0]),
        logical_pitch_um=float(phase_export["logical_pixel_pitch_um"]),
        native_pitch_um=float(phase_export["native_pixel_pitch_um"]),
    )
    if bool(phase_export.get("flip_horizontal_before_rasterization", False)):
        logical = np.fliplr(logical)
    if bool(phase_export.get("flip_vertical_before_rasterization", False)):
        logical = np.flipud(logical)
    return np.ascontiguousarray(logical, dtype=np.float32) * (2.0 * math.pi / 256.0)


def _load_amplitude(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        opened.load()
        if opened.mode != "L" or opened.size != (1024, 1024):
            raise RuntimeError(f"Amplitude BMP must be L/1024x1024: {path}")
        full = np.asarray(opened, dtype=np.float32)
    left, top, right, bottom = ACTIVE_BOUNDS
    return np.ascontiguousarray(full[top:bottom, left:right] / 255.0)


def _load_measured(path: Path) -> tuple[np.ndarray, float]:
    with Image.open(path) as opened:
        opened.load()
        if opened.mode != "L" or opened.size != (478, 478):
            raise RuntimeError(
                f"Canonical CCD must be L/478x478; got {opened.mode}/{opened.size}: {path}"
            )
        return np.asarray(opened, dtype=np.float32), 255.0


def _roi_energies(value: np.ndarray) -> list[float]:
    return [
        float(value[top:bottom, left:right].sum())
        for left, top, right, bottom in DETECTOR_BOUNDS
    ]


def _cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return None if denominator <= 1.0e-12 else float(np.dot(x, y) / denominator)


def _canonical_capture_contract(stage: Path, rows: list[dict[str, str]]) -> None:
    manifest = stage / "acquisition_logs" / "capture_manifest.csv"
    captured = _read_csv(manifest)
    if len(captured) != len(rows):
        raise RuntimeError("capture_manifest.csv count differs from samples.csv")
    required = {
        "orientation_canonicalized": "True",
        "saved_frame_orientation": "canonical_model_xy",
        "downstream_loader_flip_required": "False",
        "background_subtraction": "False",
        "per_frame_minmax_normalization": "False",
        "phase_manifest_verified": "True",
    }
    for key, expected in required.items():
        observed = {str(row.get(key, "")) for row in captured}
        if observed != {expected}:
            raise RuntimeError(
                f"Formal MNIST agreement requires {key}={expected}; got {sorted(observed)}"
            )
    geometry_hashes = {
        row.get("detector_geometry_file_sha256", "") for row in captured
    }
    if len(geometry_hashes) != 1 or not next(iter(geometry_hashes)):
        raise RuntimeError("All CCD frames must use one non-empty homography SHA-256")
    expected_amplitudes = sorted(row["amplitude_file"] for row in rows)
    observed_amplitudes = [row.get("amplitude_bmp", "") for row in captured]
    if observed_amplitudes != expected_amplitudes:
        raise RuntimeError("Captured amplitude order differs from samples.csv")


class PlayedBMPSimulator:
    def __init__(self, model_config: Path, phase: np.ndarray, device: Any) -> None:
        import torch
        from torch.nn import functional as F

        from .modeling import AngularSpectrumKSpacePropagator

        self.torch = torch
        self.F = F
        settings = load_settings(model_config)
        self.settings = settings
        self.device = device
        self.phase = torch.from_numpy(phase).to(device=device, dtype=torch.float32)
        self.propagator = AngularSpectrumKSpacePropagator(
            grid_size=settings.propagation_grid_size,
            wavelength_nm=settings.wavelength_nm,
            pixel_pitch_um=settings.logical_pixel_pitch_um,
            distance_m=settings.detector_distance_m,
            k_space_enabled=settings.k_space_enabled,
            theta_max_deg=settings.k_space_theta_max_deg,
        ).to(device)

    def __call__(self, amplitudes: np.ndarray) -> np.ndarray:
        torch, F = self.torch, self.F
        with torch.inference_mode():
            return self._forward(amplitudes)

    def _forward(self, amplitudes: np.ndarray) -> np.ndarray:
        torch, F = self.torch, self.F
        active = torch.from_numpy(amplitudes).to(self.device, dtype=torch.float32)
        modulated = active.to(torch.complex64) * torch.exp(1j * self.phase).to(
            torch.complex64
        )
        canvas = F.pad(
            modulated,
            (self.settings.canvas_guard,) * 4,
        )
        numerical = F.pad(
            canvas,
            (self.settings.propagation_guard,) * 4,
        )
        detector = self.propagator(numerical)
        guard = self.settings.propagation_guard + self.settings.canvas_guard
        field = detector[
            :,
            guard : guard + self.settings.active_size,
            guard : guard + self.settings.active_size,
        ]
        return field.abs().square().float().cpu().numpy()


def _batches(values: list[dict[str, str]], size: int) -> Iterable[list[dict[str, str]]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _finite(values: Iterable[Any]) -> np.ndarray:
    return np.asarray(
        [float(value) for value in values if value is not None and np.isfinite(value)],
        dtype=np.float64,
    )


def _aggregate(rows: list[dict[str, Any]], key: str) -> dict[str, float | int | None]:
    values = _finite(row.get(key) for row in rows)
    if values.size == 0:
        return {"count": 0, "mean": None, "median": None, "std": None, "q05": None, "q95": None}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
    }


def _configure_matplotlib() -> tuple[Any, str | None]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import font_manager, pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.0,
            "axes.titlesize": 7.0,
            "axes.labelsize": 7.0,
            "axes.linewidth": 0.6,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.5,
            "legend.frameon": False,
            "lines.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    try:
        resolved = font_manager.findfont("Arial", fallback_to_default=False)
    except ValueError:
        resolved = None
    return plt, resolved


def _save_figure(fig: Any, directory: Path, name: str) -> list[dict[str, Any]]:
    records = []
    for suffix in ("pdf", "svg", "png"):
        path = directory / f"{name}.{suffix}"
        fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        records.append(
            {
                "file": path.name,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "font": "Arial 7 pt",
            }
        )
    return records


def _display_map(value: np.ndarray) -> np.ndarray:
    source = np.clip(np.asarray(value, dtype=np.float64), 0.0, None)
    scale = float(np.quantile(source, 0.995))
    return np.clip(source / max(scale, 1.0e-12), 0.0, 1.0)


def _make_figures(
    *,
    rows: list[dict[str, Any]],
    examples: list[tuple[dict[str, Any], np.ndarray, np.ndarray]],
    output_dir: Path,
) -> dict[str, Any]:
    import torch

    plt, resolved_font = _configure_matplotlib()
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    inventory: list[dict[str, Any]] = []

    metric_names = ("pcc_full", "ssim", "cosine_similarity", "shape_nrmse")
    fig, axes = plt.subplots(
        1, 4, figsize=(12.0 * CM_TO_INCH, 5.0 * CM_TO_INCH), constrained_layout=True
    )
    for axis, name in zip(axes, metric_names):
        values = _finite(row.get(name) for row in rows)
        axis.boxplot(values, widths=0.45, showfliers=False)
        jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) else []
        axis.scatter(np.ones(len(values)) + jitter, values, s=2.5, alpha=0.35, color="#0072B2")
        axis.set_xticks([])
        axis.set_title(name.replace("_", " "))
        if name == "shape_nrmse":
            axis.set_ylabel("Lower is better")
        else:
            axis.set_ylabel("Higher is better")
    inventory.extend(_save_figure(fig, figures, "agreement_metric_distributions"))
    plt.close(fig)

    fig, axes = plt.subplots(
        len(examples),
        3,
        squeeze=False,
        figsize=(8.5 * CM_TO_INCH, 6.0 * CM_TO_INCH),
        constrained_layout=True,
    )
    for row_index, (metadata, measured, theory) in enumerate(examples):
        measured_map, theory_map = _display_map(measured), _display_map(theory)
        difference = np.abs(measured_map - theory_map)
        for column, (value, title, cmap) in enumerate(
            (
                (measured_map, "Measured", "magma"),
                (theory_map, "Simulation", "magma"),
                (difference, "|shape difference|", "viridis"),
            )
        ):
            axes[row_index, column].imshow(value, cmap=cmap, vmin=0.0, vmax=1.0)
            axes[row_index, column].set_xticks([])
            axes[row_index, column].set_yticks([])
            if row_index == 0:
                axes[row_index, column].set_title(title)
        axes[row_index, 0].set_ylabel(
            f"y={metadata['label']}, p={metadata['measured_prediction']}\n"
            f"PCC={metadata['pcc_full']:.3f}",
            fontsize=6.0,
        )
    inventory.extend(_save_figure(fig, figures, "agreement_examples"))
    plt.close(fig)

    labels = np.asarray([int(row["label"]) for row in rows])
    measured_predictions = np.asarray([int(row["measured_prediction"]) for row in rows])
    simulated_predictions = np.asarray([int(row["simulation_prediction"]) for row in rows])
    measured_confusion = np.zeros((4, 4), dtype=np.int64)
    simulated_confusion = np.zeros((4, 4), dtype=np.int64)
    for label, measured_prediction, simulated_prediction in zip(
        labels, measured_predictions, simulated_predictions
    ):
        measured_confusion[label, measured_prediction] += 1
        simulated_confusion[label, simulated_prediction] += 1
    fig, axes = plt.subplots(
        1, 2, figsize=(8.0 * CM_TO_INCH, 5.0 * CM_TO_INCH), constrained_layout=True
    )
    for axis, matrix, title in (
        (axes[0], simulated_confusion, "Simulation"),
        (axes[1], measured_confusion, "Measured CCD"),
    ):
        normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
        for true_id in range(4):
            for prediction in range(4):
                axis.text(prediction, true_id, str(matrix[true_id, prediction]), ha="center", va="center", fontsize=6)
        axis.set_xticks(range(4))
        axis.set_yticks(range(4))
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_title(title)
    inventory.extend(_save_figure(fig, figures, "simulation_measured_confusion"))
    plt.close(fig)
    return {"resolved_arial_font": resolved_font, "files": inventory}


def evaluate_agreement(
    *,
    stage_dir: Path,
    bundle_root: Path,
    output_dir: Path | None = None,
    device: str = "auto",
    batch_size: int = 4,
) -> dict[str, Any]:
    stage = stage_dir.expanduser().resolve()
    root = bundle_root.expanduser().resolve()
    stage_report = validate_session(stage)
    rows = _read_csv(stage / "samples.csv")
    _canonical_capture_contract(stage, rows)
    contract = _read_json(stage / "stage_contract.json")
    phase_files = sorted((stage / "phase_to_play").glob("*.bmp"))
    phase = decode_played_phase(phase_files[0], contract)
    selected_device = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    )
    simulator = PlayedBMPSimulator(
        root / "payload" / "model" / "lab_model_config.yaml",
        phase,
        selected_device,
    )
    results: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(_batches(rows, max(1, int(batch_size)))):
        amplitudes = np.stack(
            [_load_amplitude(stage / "amplitude_to_play" / row["amplitude_file"]) for row in batch]
        )
        theories = simulator(amplitudes)
        for row, theory in zip(batch, theories):
            measured, saturation_value = _load_measured(
                stage / "ccd_captured" / f"{row['key']}.png"
            )
            metrics = pair_metrics(
                measured,
                theory,
                domain="linear",
                energy_gain=1.0,
                saturation_value=saturation_value,
                signal_energy_fraction=0.99,
                relative_clip=20.0,
                log_compression=1.0,
            )
            measured_energies = _roi_energies(measured)
            theory_energies = _roi_energies(theory)
            results.append(
                {
                    "key": row["key"],
                    "label": int(row["label"]),
                    "measured_prediction": int(np.argmax(measured_energies)),
                    "simulation_prediction": int(np.argmax(theory_energies)),
                    "measured_correct": int(np.argmax(measured_energies)) == int(row["label"]),
                    "simulation_correct": int(np.argmax(theory_energies)) == int(row["label"]),
                    **metrics,
                    "cosine_similarity": _cosine(measured, theory),
                    **{f"measured_roi_energy_{index}": value for index, value in enumerate(measured_energies)},
                    **{f"simulation_roi_energy_{index}": value for index, value in enumerate(theory_energies)},
                }
            )
        print(f"[mnist4 agreement] {min((batch_index + 1) * batch_size, len(rows))}/{len(rows)}")

    raw_ratios = _finite(row["energy_ratio_raw"] for row in results)
    energy_gain = float(np.median(raw_ratios)) if raw_ratios.size else 1.0
    for row in results:
        raw = row.get("energy_ratio_raw")
        row["energy_ratio_calibrated"] = (
            None if raw is None or energy_gain <= 0.0 else float(raw) / energy_gain
        )

    destination = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else stage / "simulation_agreement"
    )
    destination.mkdir(parents=True, exist_ok=True)
    _write_csv(destination / "per_sample_agreement.csv", results)

    ranked = sorted(results, key=lambda row: float(row.get("pcc_full") or -2.0))
    selected_rows = [ranked[0], ranked[len(ranked) // 2], ranked[-1]]
    examples: list[tuple[dict[str, Any], np.ndarray, np.ndarray]] = []
    for selected in selected_rows:
        source = next(row for row in rows if row["key"] == selected["key"])
        theory = simulator(
            np.stack([_load_amplitude(stage / "amplitude_to_play" / source["amplitude_file"])])
        )[0]
        measured, _ = _load_measured(stage / "ccd_captured" / f"{source['key']}.png")
        examples.append((selected, measured, theory))

    figures = _make_figures(rows=results, examples=examples, output_dir=destination)
    metric_keys = (
        "pcc_full",
        "pcc_signal",
        "ssim",
        "shape_nrmse",
        "cosine_similarity",
        "energy_ratio_raw",
        "energy_ratio_calibrated",
        "centroid_distance_px",
        "outside_energy_fraction",
        "saturation_fraction",
    )
    measured_accuracy = float(np.mean([row["measured_correct"] for row in results]))
    simulation_accuracy = float(np.mean([row["simulation_correct"] for row in results]))
    summary = {
        "task": "MNIST digits 0-3, single 10 cm optical layer",
        "stage": stage_report,
        "device": str(selected_device),
        "samples": len(results),
        "measured_raw_four_roi_accuracy": measured_accuracy,
        "simulation_raw_four_roi_accuracy_from_played_bmps": simulation_accuracy,
        "prediction_agreement_fraction": float(
            np.mean(
                [row["measured_prediction"] == row["simulation_prediction"] for row in results]
            )
        ),
        "median_measured_to_simulation_energy_gain": energy_gain,
        "metrics": {key: _aggregate(results, key) for key in metric_keys},
        "classification_contract": {
            "ccd": "canonical 478x478 uint8 intensity",
            "score": "four untouched 59x59 ROI sums then argmax",
            "normalization": False,
            "nonlinearity": False,
            "background_subtraction": False,
            "resize_after_homography": False,
        },
        "agreement_metric_contract": {
            "pcc_and_cosine": "scale insensitive, evaluated on full linear intensity",
            "ssim": "per-frame mean-scaled/clipped shape map; analysis only",
            "shape_nrmse": "unit-energy shape comparison; analysis only",
            "energy_ratio": "reported separately before/after one dataset-level median gain",
            "signal_mask": "simulation pixels containing 99% of simulated energy",
        },
        "phase_reconstruction": contract["phase_export"],
        "figures": figures,
    }
    _write_json(destination / "agreement_summary.json", summary)
    _write_json(
        destination / "output_inventory.json",
        {
            "files": [
                {
                    "path": path.relative_to(destination).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in sorted(destination.rglob("*"))
                if path.is_file() and path.name != "output_inventory.json"
            ]
        },
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-dir", required=True)
    parser.add_argument(
        "--bundle-root",
        default=str(Path(__file__).resolve().parents[1] / "lab_qwen" / "mnist4"),
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)
    report = evaluate_agreement(
        stage_dir=Path(args.stage_dir),
        bundle_root=Path(args.bundle_root),
        output_dir=None if args.output_dir is None else Path(args.output_dir),
        device=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
