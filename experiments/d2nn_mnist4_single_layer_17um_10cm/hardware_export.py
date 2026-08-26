from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from experiments.hardware_sdk.workflows.reconstruct_slm import (
    encode_active_phase,
    physical_pitch_nearest,
    place_at_center,
)

from .io_utils import sha256, write_csv, write_json
from .hardware_profiles import (
    DEMO_PROFILE,
    PHASE_FILENAME,
    formal_profile_name,
    select_demo_topk,
    select_formal_fixed_random,
)
from .modeling import SingleLayerMNIST4D2NN
from .settings import Settings


def _save_checked_bmp(
    array: np.ndarray, path: Path, size_wh: tuple[int, int]
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8), mode="L").save(
        path, format="BMP"
    )
    with Image.open(path) as image:
        if image.format != "BMP" or image.mode != "L" or image.size != size_wh:
            raise RuntimeError(
                f"Invalid BMP contract for {path}: "
                f"{image.format}/{image.mode}/{image.size}"
            )
    return sha256(path)


def _full_amplitude_frame(
    active_amplitude: np.ndarray, settings: Settings
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    optical = np.rint(np.clip(active_amplitude, 0.0, 1.0) * 255.0).astype(
        np.uint8
    )
    encoded = 255 - optical if settings.amplitude_invert_before_export else optical
    width, height = settings.amplitude_slm_size_wh
    background = 255 if settings.amplitude_invert_before_export else 0
    canvas = Image.new("L", (width, height), color=background)
    active = Image.fromarray(encoded, mode="L")
    requested_x, requested_y = settings.amplitude_slm_center_xy
    left = int(math.floor(requested_x - active.width / 2.0 + 0.5))
    top = int(math.floor(requested_y - active.height / 2.0 + 0.5))
    bounds = (left, top, left + active.width, top + active.height)
    if left < 0 or top < 0 or bounds[2] > width or bounds[3] > height:
        raise ValueError(
            f"Amplitude active bounds {bounds} exceed SLM {(width, height)}"
        )
    canvas.paste(active, (left, top))
    return np.asarray(canvas, dtype=np.uint8), bounds


@torch.no_grad()
def _collect_test_samples(
    model: SingleLayerMNIST4D2NN,
    dataset: torch.utils.data.Dataset,
    settings: Settings,
    device: torch.device,
) -> dict[int, list[dict[str, Any]]]:
    """Evaluate every candidate once without imposing a selection policy."""

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
        images = images.to(device, non_blocking=True)
        output = model(images)
        fractions = output["detector_fraction"].detach().cpu()
        predictions = fractions.argmax(dim=1)
        for local_index, target in enumerate(targets):
            label = int(target)
            scores = fractions[local_index]
            sorted_scores = scores.sort(descending=True).values
            candidates[label].append(
                {
                    "dataset_index": offset + local_index,
                    "label": label,
                    "prediction": int(predictions[local_index]),
                    "correct": int(predictions[local_index]) == label,
                    "target_fraction": float(scores[label]),
                    "margin": float(sorted_scores[0] - sorted_scores[1]),
                    "detector_fractions": [float(value) for value in scores],
                }
            )
        offset += len(targets)
    missing = [label for label, values in candidates.items() if not values]
    if missing:
        raise RuntimeError(f"The test dataset contains no samples for classes {missing}")
    return candidates


def _clear_generated_bmps(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.glob("*.bmp"):
        if path.is_file():
            path.unlink()


def _write_profile(
    *,
    profile: str,
    selection_policy: str,
    selected: dict[int, list[dict[str, Any]]],
    model: SingleLayerMNIST4D2NN,
    test_dataset: torch.utils.data.Dataset,
    settings: Settings,
    output_dir: Path,
    phase_canvas: np.ndarray,
    phase_sha256: str,
    random_seed: int | None,
    suitable_for_accuracy_reporting: bool,
) -> tuple[list[dict[str, Any]], tuple[int, int, int, int]]:
    stage_dir = output_dir / profile
    amplitude_dir = stage_dir / "amplitude_to_play"
    phase_dir = stage_dir / "phase_to_play"
    _clear_generated_bmps(amplitude_dir)
    _clear_generated_bmps(phase_dir)
    stage_phase = phase_dir / PHASE_FILENAME
    copied_phase_sha = _save_checked_bmp(
        phase_canvas, stage_phase, settings.phase_slm_size_wh
    )
    if copied_phase_sha != phase_sha256:
        raise RuntimeError("The duplicated stage phase mask changed during export")

    rows: list[dict[str, Any]] = []
    amplitude_bounds: tuple[int, int, int, int] | None = None
    for label in settings.classes:
        for selection_rank, item in enumerate(selected[label]):
            image, target = test_dataset[item["dataset_index"]]
            active = model.prepare_active_amplitude(image.unsqueeze(0)).squeeze(0)
            full, bounds = _full_amplitude_frame(active.cpu().numpy(), settings)
            amplitude_bounds = bounds
            key = (
                f"{profile}_i{item['dataset_index']:05d}_"
                f"y{int(target)}_r{selection_rank:03d}"
            )
            path = amplitude_dir / f"{key}.bmp"
            digest = _save_checked_bmp(full, path, settings.amplitude_slm_size_wh)
            rows.append(
                {
                    "key": key,
                    "profile": profile,
                    "selection_policy": selection_policy,
                    "selection_seed": "" if random_seed is None else int(random_seed),
                    "selection_rank_within_class": selection_rank,
                    "amplitude_file": path.name,
                    "dataset_index": item["dataset_index"],
                    "label": int(target),
                    "simulation_prediction": item["prediction"],
                    "simulation_correct": item["correct"],
                    "simulation_target_fraction": item["target_fraction"],
                    "simulation_margin": item["margin"],
                    "detector_fraction_0": item["detector_fractions"][0],
                    "detector_fraction_1": item["detector_fractions"][1],
                    "detector_fraction_2": item["detector_fractions"][2],
                    "detector_fraction_3": item["detector_fractions"][3],
                    "amplitude_sha256": digest,
                    "phase_file": PHASE_FILENAME,
                    "phase_sha256": phase_sha256,
                }
            )
    if amplitude_bounds is None:
        raise RuntimeError(f"No amplitude samples were written for profile {profile}")
    write_csv(stage_dir / "samples.csv", rows)
    stage_contract = {
        "schema_version": 1,
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
    }
    write_json(stage_dir / "stage_contract.json", stage_contract)
    warning = (
        "This profile intentionally selects easy, high-margin examples and must "
        "not be reported as an unbiased hardware accuracy."
        if not suitable_for_accuracy_reporting
        else "This profile is fixed-random within each true class and does not use model predictions during selection."
    )
    (stage_dir / "README.md").write_text(
        f"""# {profile}

{warning}

Keep `{PHASE_FILENAME}` loaded on the phase SLM, play every BMP under
`amplitude_to_play/`, and save each CCD frame with the same basename. The
laboratory pipeline writes captures to `ccd_captured/` and evaluates them
against `samples.csv`.
""",
        encoding="utf-8",
    )
    return rows, amplitude_bounds


def _write_detector_definition(output_dir: Path, settings: Settings) -> None:
    detector_preview = np.zeros(
        (settings.active_size, settings.active_size), dtype=np.uint8
    )
    detector_rows = []
    for label, (left, top, right, bottom) in enumerate(settings.detector_bounds()):
        detector_preview[top:bottom, left:right] = 64 + label * 48
        detector_rows.append(
            {
                "class": label,
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "center_edge_x": (left + right) / 2.0,
                "center_edge_y": (top + bottom) / 2.0,
            }
        )
    Image.fromarray(detector_preview, mode="L").save(
        output_dir / "detector_roi_478.png"
    )
    write_csv(output_dir / "detector_regions.csv", detector_rows)


def _write_lab_model_config(output_dir: Path, settings: Settings) -> None:
    """Write a standalone merged YAML-compatible JSON for CCD evaluation."""

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
            "detector_distance_m": settings.detector_distance_m,
            "phase": {
                "parameterization": settings.phase_parameterization,
                "init": settings.phase_init,
            },
        },
        "detector": {"size": settings.detector_size},
        "loss": {
            "eps": settings.loss_eps,
            "template_mse_weight": settings.template_mse_loss_weight,
            "detector_ce_weight": settings.detector_ce_loss_weight,
        },
        "training": {
            "optimizer": settings.optimizer,
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
                "invert_before_export": settings.amplitude_invert_before_export,
            },
            "phase_slm": {
                "size_wh": list(settings.phase_slm_size_wh),
                "pixel_pitch_um": settings.phase_slm_pixel_pitch_um,
                "center_xy": list(settings.phase_slm_center_xy),
                "flip_vertical": settings.phase_flip_vertical,
                "flip_horizontal": settings.phase_flip_horizontal,
            },
            "ccd": {"target_size": settings.ccd_target_size},
            "demo_samples_per_class": settings.demo_samples_per_class,
            "evaluation_samples_per_class": settings.evaluation_samples_per_class,
            "export_subdir": settings.hardware_export_subdir,
        },
        "device": "cpu",
        "output_dir": ".",
    }
    # JSON is a strict subset of YAML and avoids carrying a base_config path
    # that would be invalid on the laboratory computer.
    (output_dir / "lab_model_config.yaml").write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def export_hardware_bundle(
    model: SingleLayerMNIST4D2NN,
    test_dataset: torch.utils.data.Dataset,
    settings: Settings,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    phase = model.phase().detach().cpu().numpy()
    logical_phase = encode_active_phase(phase)
    if settings.phase_flip_vertical:
        logical_phase = np.flipud(logical_phase)
    if settings.phase_flip_horizontal:
        logical_phase = np.fliplr(logical_phase)
    logical_phase = np.ascontiguousarray(logical_phase)
    native_phase = physical_pitch_nearest(
        logical_phase,
        logical_pixel_pitch_um=settings.logical_pixel_pitch_um,
        slm_pixel_pitch_um=settings.phase_slm_pixel_pitch_um,
    )
    phase_canvas_image, phase_bounds, actual_phase_center = place_at_center(
        Image.fromarray(native_phase, mode="L"),
        slm_size_wh=settings.phase_slm_size_wh,
        center_xy=settings.phase_slm_center_xy,
    )
    phase_canvas = np.asarray(phase_canvas_image, dtype=np.uint8)
    canonical_phase = output_dir / "phase_to_play" / PHASE_FILENAME
    _clear_generated_bmps(canonical_phase.parent)
    phase_sha = _save_checked_bmp(
        phase_canvas, canonical_phase, settings.phase_slm_size_wh
    )

    candidates = _collect_test_samples(model, test_dataset, settings, device)
    formal_profile = formal_profile_name(settings.evaluation_samples_per_class)
    demo = select_demo_topk(
        candidates,
        settings.classes,
        samples_per_class=settings.demo_samples_per_class,
    )
    formal = select_formal_fixed_random(
        candidates,
        settings.classes,
        samples_per_class=settings.evaluation_samples_per_class,
        seed=settings.random_seed,
        # A deliberately limited smoke dataset may export fewer samples; a
        # full formal dataset must never silently claim an incomplete profile.
        require_full_count=settings.test_limit is None,
    )
    demo_rows, amplitude_bounds = _write_profile(
        profile=DEMO_PROFILE,
        selection_policy="simulation_correct_then_descending_margin",
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
    formal_rows, formal_amplitude_bounds = _write_profile(
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
    if formal_amplitude_bounds != amplitude_bounds:
        raise RuntimeError("Amplitude placement changed between export profiles")
    _write_detector_definition(output_dir, settings)
    _write_lab_model_config(output_dir, settings)

    contract = {
        "schema_version": 2,
        "model": "single-layer phase-only MNIST-4 D2NN",
        "classes": list(settings.classes),
        "wavelength_nm": settings.wavelength_nm,
        "phase_to_ccd_distance_cm": settings.detector_distance_m * 100.0,
        "input_phase_relation": (
            "4F co-planar; no simulated free-space propagation before phase"
        ),
        "logical_geometry": {
            "canvas_size": settings.canvas_size,
            "propagation_grid_size": settings.propagation_grid_size,
            "active_size": settings.active_size,
            "input_size": settings.input_size,
            "detector_bounds_xyxy": [
                list(value) for value in settings.detector_bounds()
            ],
        },
        "amplitude_slm": {
            "size_wh": list(settings.amplitude_slm_size_wh),
            "pixel_pitch_um": settings.amplitude_slm_pixel_pitch_um,
            "center_xy": list(settings.amplitude_slm_center_xy),
            "active_bounds_xyxy": list(amplitude_bounds),
            "invert_before_export": settings.amplitude_invert_before_export,
            "outside_active_value_uint8": (
                255 if settings.amplitude_invert_before_export else 0
            ),
            "bright_value_uint8": (
                0 if settings.amplitude_invert_before_export else 255
            ),
            "dark_value_uint8": (
                255 if settings.amplitude_invert_before_export else 0
            ),
        },
        "phase_slm": {
            "file": PHASE_FILENAME,
            "sha256": phase_sha,
            "size_wh": list(settings.phase_slm_size_wh),
            "native_pixel_pitch_um": settings.phase_slm_pixel_pitch_um,
            "logical_pixel_pitch_um": settings.logical_pixel_pitch_um,
            "native_active_size_hw": list(native_phase.shape),
            "active_bounds_xyxy": list(phase_bounds),
            "actual_center_edge_xy": list(actual_phase_center),
            "flip_vertical_before_export": settings.phase_flip_vertical,
            "flip_horizontal_before_export": settings.phase_flip_horizontal,
            "phase_parameterization": "2*pi*sigmoid(raw_phase)",
        },
        "ccd": {
            "roi_source": "derive from the 4-focus Fresnel calibration",
            "target_size": settings.ccd_target_size,
            "normalization": (
                "detector energies divided by total frame energy; global gain invariant"
            ),
            "background_subtraction": False,
            "flip": (
                "apply the experimentally measured Fresnel correspondence before classification"
            ),
        },
        "profiles": {
            DEMO_PROFILE: {
                "sample_count": len(demo_rows),
                "manifest": f"{DEMO_PROFILE}/samples.csv",
                "accuracy_reporting": False,
            },
            formal_profile: {
                "sample_count": len(formal_rows),
                "requested_samples_per_class": settings.evaluation_samples_per_class,
                "actual_samples_per_class": {
                    str(label): len(formal[label]) for label in settings.classes
                },
                "manifest": f"{formal_profile}/samples.csv",
                "selection_seed": settings.random_seed,
                "accuracy_reporting": True,
            },
        },
        "sample_count": len(demo_rows) + len(formal_rows),
        "demo_sample_count": len(demo_rows),
        "formal_sample_count": len(formal_rows),
    }
    write_json(output_dir / "hardware_contract.json", contract)
    (output_dir / "README_LAB.md").write_text(
        f"""# MNIST-4 10 cm laboratory payload

Hardware polarity is **255 = white/transmissive, 0 = black/blocking**. No
amplitude inversion is applied when `invert_before_export` is false.

- `{DEMO_PROFILE}/`: easy high-margin examples for alignment and a quick demo.
  Its accuracy is selection-biased and must not be reported as test accuracy.
- `{formal_profile}/`: {settings.evaluation_samples_per_class} fixed-random samples from each
  true class (seed {settings.random_seed}); model predictions never participate
  in selection. Use only this profile for hardware accuracy.

Both directories are complete acquisition stages: each contains exactly one
10 cm phase BMP, amplitude BMPs, and `samples.csv`. Use `lab_pipeline.py` to
validate the devices, acquire same-name CCD frames, and compute four-region
accuracy. No background subtraction is performed.
""",
        encoding="utf-8",
    )
    return contract
