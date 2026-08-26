from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from experiments.d2nn_mnist4_single_layer_17um_10cm.hardware_export import (
    _full_amplitude_frame,
    _save_checked_bmp,
)
from experiments.d2nn_mnist4_single_layer_17um_10cm.hardware_profiles import (
    DEMO_PROFILE,
    formal_profile_name,
    select_demo_topk,
    select_formal_fixed_random,
)
from experiments.hardware_sdk.workflows.reconstruct_slm import (
    encode_active_phase,
    physical_pitch_nearest,
    place_at_center,
)

from .io_utils import write_csv, write_json
from .modeling import RobustRawCCDMNIST4D2NN
from .settings import V2Settings


PHASE_FILENAME = "mnist4_single_layer_17um_10cm_v2.bmp"


def _clear_bmps(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.glob("*.bmp"):
        path.unlink()


@torch.no_grad()
def _collect_test_samples(
    model: RobustRawCCDMNIST4D2NN,
    dataset: torch.utils.data.Dataset,
    settings: V2Settings,
    device: torch.device,
) -> dict[int, list[dict[str, Any]]]:
    loader = DataLoader(
        dataset,
        batch_size=settings.inference_batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    model.eval()
    candidates = {label: [] for label in settings.classes}
    offset = 0
    for images, targets in loader:
        output = model(images.to(device, non_blocking=True))
        energies = output["detector_energy"]
        if not isinstance(energies, torch.Tensor):
            raise TypeError("detector_energy must be a tensor")
        energies = energies.detach().cpu()
        predictions = energies.argmax(dim=1)
        for local_index, target in enumerate(targets):
            label = int(target)
            scores = energies[local_index]
            sorted_scores = scores.sort(descending=True).values
            candidates[label].append(
                {
                    "dataset_index": offset + local_index,
                    "label": label,
                    "prediction": int(predictions[local_index]),
                    "correct": int(predictions[local_index]) == label,
                    "target_energy": float(scores[label]),
                    "margin": float(sorted_scores[0] - sorted_scores[1]),
                    "detector_energies": [float(value) for value in scores],
                }
            )
        offset += len(targets)
    missing = [label for label, values in candidates.items() if not values]
    if missing:
        raise RuntimeError(f"Test dataset contains no samples for classes {missing}")
    return candidates


def _write_profile(
    *,
    profile: str,
    selection_policy: str,
    selected: dict[int, list[dict[str, Any]]],
    model: RobustRawCCDMNIST4D2NN,
    test_dataset: torch.utils.data.Dataset,
    settings: V2Settings,
    output_dir: Path,
    phase_canvas: np.ndarray,
    phase_sha256: str,
    random_seed: int | None,
    suitable_for_accuracy_reporting: bool,
) -> tuple[list[dict[str, Any]], tuple[int, int, int, int]]:
    stage = output_dir / profile
    amplitude_dir = stage / "amplitude_to_play"
    phase_dir = stage / "phase_to_play"
    _clear_bmps(amplitude_dir)
    _clear_bmps(phase_dir)
    stage_phase = phase_dir / PHASE_FILENAME
    copied_sha = _save_checked_bmp(
        phase_canvas, stage_phase, settings.phase_slm_size_wh
    )
    if copied_sha != phase_sha256:
        raise RuntimeError("Stage phase copy changed during export")

    rows: list[dict[str, Any]] = []
    amplitude_bounds: tuple[int, int, int, int] | None = None
    for label in settings.classes:
        for rank, item in enumerate(selected[label]):
            image, target = test_dataset[item["dataset_index"]]
            active = model.prepare_active_amplitude(image.unsqueeze(0)).squeeze(0)
            full, bounds = _full_amplitude_frame(active.cpu().numpy(), settings)
            amplitude_bounds = bounds
            key = f"{profile}_i{item['dataset_index']:05d}_y{int(target)}_r{rank:03d}"
            path = amplitude_dir / f"{key}.bmp"
            amplitude_sha = _save_checked_bmp(
                full, path, settings.amplitude_slm_size_wh
            )
            row = {
                "key": key,
                "profile": profile,
                "selection_policy": selection_policy,
                "selection_seed": "" if random_seed is None else random_seed,
                "selection_rank_within_class": rank,
                "amplitude_file": path.name,
                "dataset_index": item["dataset_index"],
                "label": int(target),
                "simulation_prediction": item["prediction"],
                "simulation_correct": item["correct"],
                "simulation_target_energy": item["target_energy"],
                "simulation_raw_energy_margin": item["margin"],
                "amplitude_sha256": amplitude_sha,
                "phase_file": PHASE_FILENAME,
                "phase_sha256": phase_sha256,
            }
            for index, energy in enumerate(item["detector_energies"]):
                row[f"detector_raw_energy_{index}"] = energy
            rows.append(row)
    if amplitude_bounds is None:
        raise RuntimeError(f"No amplitudes were written for profile {profile}")
    write_csv(stage / "samples.csv", rows)
    write_json(
        stage / "stage_contract.json",
        {
            "schema_version": 2,
            "profile": profile,
            "selection_policy": selection_policy,
            "selection_seed": random_seed,
            "samples": len(rows),
            "samples_per_class": {
                str(label): len(selected[label]) for label in settings.classes
            },
            "suitable_for_accuracy_reporting": suitable_for_accuracy_reporting,
            "phase_file": f"phase_to_play/{PHASE_FILENAME}",
            "phase_sha256": phase_sha256,
            "amplitude_directory": "amplitude_to_play",
            "capture_directory": "ccd_captured",
            "manifest": "samples.csv",
            "ccd_readout": "raw_region_sums_no_normalization",
        },
    )
    warning = (
        "Biased simulation-selected demonstration; never report its success rate."
        if not suitable_for_accuracy_reporting
        else "Fixed-random samples within true class; model output does not affect selection."
    )
    (stage / "README.md").write_text(
        f"# {profile}\n\n{warning}\n\n"
        "Capture exactly 478x478 8-bit raw CCD frames. Do not normalize, resize, "
        "apply nonlinear compression, or subtract an unmeasured background.\n",
        encoding="utf-8",
    )
    return rows, amplitude_bounds


def _write_detector_definition(output_dir: Path, settings: V2Settings) -> None:
    preview = np.zeros((settings.active_size, settings.active_size), dtype=np.uint8)
    rows = []
    for label, (left, top, right, bottom) in enumerate(settings.detector_bounds()):
        preview[top:bottom, left:right] = 64 + 48 * label
        rows.append(
            {
                "class": label,
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "width": right - left,
                "height": bottom - top,
                "center_edge_x": (left + right) / 2.0,
                "center_edge_y": (top + bottom) / 2.0,
            }
        )
    Image.fromarray(preview, mode="L").save(output_dir / "detector_roi_478.png")
    write_csv(output_dir / "detector_regions.csv", rows)


def _write_lab_model_config(output_dir: Path, settings: V2Settings) -> None:
    value = {
        "dataset": {
            "root": ".",
            "classes": list(settings.classes),
            "download": False,
            "val_fraction": settings.val_fraction,
        },
        "optics": {
            "wavelength_nm": settings.wavelength_nm,
            "logical_pixel_pitch_um": settings.logical_pixel_pitch_um,
            "canvas_size": settings.canvas_size,
            "propagation_grid_size": settings.propagation_grid_size,
            "active_size": settings.active_size,
            "input_size": settings.input_size,
            "input_content_size": settings.input_content_size,
            "detector_distance_m": settings.detector_distance_m,
            "phase": {"parameterization": "sigmoid", "init": "zeros"},
            "k_space": {
                "enabled": settings.k_space_enabled,
                "theta_max_deg": settings.k_space_theta_max_deg,
            },
        },
        "detector": {
            "size": settings.detector_size,
            "mapping_mode": settings.detector_mapping_mode,
            "reference_grid_size": settings.detector_reference_grid_size,
            "reference_intervals": [
                list(value) for value in settings.detector_reference_intervals
            ],
            "reference_pixel_pitch_um": (
                settings.detector_reference_pixel_pitch_um
            ),
            "reference_distance_m": settings.detector_reference_distance_m,
        },
        "robustness": {
            "enabled": settings.robustness_enabled,
            "probability": settings.robustness_probability,
            "warmup_epochs": settings.robustness_warmup_epochs,
            "input_shift_max_px": settings.input_shift_max_px,
            "phase_shift_max_px": settings.phase_shift_max_px,
            "pre_ccd_shift_max_px": settings.pre_ccd_shift_max_px,
        },
        "loss": {
            "mode": settings.loss_mode,
            "notebook_full_plane_mse_scale": (
                settings.notebook_full_plane_mse_scale
            ),
            "eps": settings.loss_eps,
            "template_mse_weight": settings.template_mse_loss_weight,
            "detector_ce_weight": settings.detector_ce_loss_weight,
            "target_region_mse_weight": settings.target_region_mse_weight,
            "background_mse_weight": settings.background_mse_weight,
        },
        "training": {
            "optimizer": "adam",
            "phase_learning_rate": settings.phase_learning_rate,
            "min_learning_rate": settings.min_learning_rate,
            "epochs": settings.epochs,
            "batch_size": settings.batch_size,
            "inference_batch_size": settings.inference_batch_size,
            "num_workers": 0,
            "random_seed": settings.random_seed,
            "gradient_clip_norm": settings.gradient_clip_norm,
            "log_interval_batches": settings.log_interval_batches,
        },
        "hardware": {
            "amplitude_slm": {
                "size_wh": list(settings.amplitude_slm_size_wh),
                "pixel_pitch_um": settings.amplitude_slm_pixel_pitch_um,
                "center_xy": list(settings.amplitude_slm_center_xy),
                "invert_before_export": False,
            },
            "phase_slm": {
                "size_wh": list(settings.phase_slm_size_wh),
                "pixel_pitch_um": settings.phase_slm_pixel_pitch_um,
                "center_xy": list(settings.phase_slm_center_xy),
                "flip_vertical": settings.phase_flip_vertical,
                "flip_horizontal": settings.phase_flip_horizontal,
            },
            "ccd": {"target_size": 478, "postprocess": "none_raw_linear"},
            "demo_samples_per_class": settings.demo_samples_per_class,
            "evaluation_samples_per_class": settings.evaluation_samples_per_class,
            "export_subdir": settings.hardware_export_subdir,
        },
        "device": "cpu",
        "output_dir": ".",
    }
    (output_dir / "lab_model_config.yaml").write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


@torch.no_grad()
def export_hardware_bundle(
    model: RobustRawCCDMNIST4D2NN,
    test_dataset: torch.utils.data.Dataset,
    settings: V2Settings,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logical_phase = encode_active_phase(model.phase().detach().cpu().numpy())
    if settings.phase_flip_vertical:
        logical_phase = np.flipud(logical_phase)
    if settings.phase_flip_horizontal:
        logical_phase = np.fliplr(logical_phase)
    native_phase = physical_pitch_nearest(
        np.ascontiguousarray(logical_phase),
        logical_pixel_pitch_um=settings.logical_pixel_pitch_um,
        slm_pixel_pitch_um=settings.phase_slm_pixel_pitch_um,
    )
    phase_image, phase_bounds, actual_phase_center = place_at_center(
        Image.fromarray(native_phase, mode="L"),
        slm_size_wh=settings.phase_slm_size_wh,
        center_xy=settings.phase_slm_center_xy,
    )
    phase_canvas = np.asarray(phase_image, dtype=np.uint8)
    canonical_phase = output_dir / "phase_to_play" / PHASE_FILENAME
    _clear_bmps(canonical_phase.parent)
    phase_sha = _save_checked_bmp(
        phase_canvas, canonical_phase, settings.phase_slm_size_wh
    )

    candidates = _collect_test_samples(model, test_dataset, settings, device)
    formal_profile = formal_profile_name(settings.evaluation_samples_per_class)
    demo = select_demo_topk(
        candidates, settings.classes, settings.demo_samples_per_class
    )
    formal = select_formal_fixed_random(
        candidates,
        settings.classes,
        samples_per_class=settings.evaluation_samples_per_class,
        seed=settings.random_seed,
        require_full_count=settings.test_limit is None,
    )
    demo_rows, amplitude_bounds = _write_profile(
        profile=DEMO_PROFILE,
        selection_policy="simulation_correct_then_descending_raw_energy_margin",
        selected=demo,
        model=model,
        test_dataset=test_dataset,
        settings=settings,
        output_dir=output_dir,
        phase_canvas=phase_canvas,
        phase_sha256=phase_sha,
        random_seed=None,
        suitable_for_accuracy_reporting=False,
    )
    formal_rows, formal_bounds = _write_profile(
        profile=formal_profile,
        selection_policy="fixed_random_within_true_class_without_model_filtering",
        selected=formal,
        model=model,
        test_dataset=test_dataset,
        settings=settings,
        output_dir=output_dir,
        phase_canvas=phase_canvas,
        phase_sha256=phase_sha,
        random_seed=settings.random_seed,
        suitable_for_accuracy_reporting=True,
    )
    if formal_bounds != amplitude_bounds:
        raise RuntimeError("Amplitude placement changed between profiles")
    _write_detector_definition(output_dir, settings)
    _write_lab_model_config(output_dir, settings)
    contract = {
        "schema_version": 3,
        "model": "single-layer robust raw-CCD MNIST-4 D2NN v2",
        "wavelength_nm": settings.wavelength_nm,
        "phase_to_ccd_distance_cm": 100.0 * settings.detector_distance_m,
        "logical_geometry": {
            "canvas_size": settings.canvas_size,
            "propagation_grid_size": settings.propagation_grid_size,
            "active_ccd_size": settings.active_size,
            "detector_bounds_xyxy": [
                list(value) for value in settings.detector_bounds()
            ],
            "notebook_reference_grid": settings.detector_reference_grid_size,
            "notebook_reference_intervals": [
                list(value) for value in settings.detector_reference_intervals
            ],
        },
        "k_space": {
            "enabled": settings.k_space_enabled,
            "theta_max_deg": settings.k_space_theta_max_deg,
            "pass_fraction": model.propagator.pass_fraction,
        },
        "training_misalignment": {
            "cardinal_only": True,
            "input_shift_max_px": settings.input_shift_max_px,
            "phase_shift_max_px": settings.phase_shift_max_px,
            "pre_ccd_shift_max_px": settings.pre_ccd_shift_max_px,
        },
        "ccd_postprocess": "none: raw region sums only",
        "background_subtraction": False,
        "phase_slm": {
            "file": PHASE_FILENAME,
            "sha256": phase_sha,
            "size_wh": list(settings.phase_slm_size_wh),
            "active_bounds_xyxy": list(phase_bounds),
            "actual_center_edge_xy": list(actual_phase_center),
            "flip_vertical_before_export": settings.phase_flip_vertical,
        },
        "amplitude_slm": {
            "size_wh": list(settings.amplitude_slm_size_wh),
            "active_bounds_xyxy": list(amplitude_bounds),
            "invert_before_export": False,
            "bright_value_uint8": 255,
            "dark_value_uint8": 0,
        },
        "profiles": {
            DEMO_PROFILE: {
                "sample_count": len(demo_rows),
                "accuracy_reporting": False,
            },
            formal_profile: {
                "sample_count": len(formal_rows),
                "accuracy_reporting": True,
            },
        },
        "sample_count": len(demo_rows) + len(formal_rows),
    }
    write_json(output_dir / "hardware_contract.json", contract)
    (output_dir / "README_LAB.md").write_text(
        "# MNIST-4 v2 raw CCD payload\n\n"
        "The CCD contract is exactly 478x478 8-bit raw intensity. After capture, "
        "use raw detector-region sums only. Do not resize, normalize, apply a "
        "nonlinearity, or subtract an unmeasured background.\n",
        encoding="utf-8",
    )
    return contract
