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
from .modeling import SingleLayerMNIST4D2NN
from .settings import Settings


def _save_checked_bmp(array: np.ndarray, path: Path, size_wh: tuple[int, int]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8), mode="L").save(
        path, format="BMP"
    )
    with Image.open(path) as image:
        if image.format != "BMP" or image.mode != "L" or image.size != size_wh:
            raise RuntimeError(
                f"Invalid BMP contract for {path}: {image.format}/{image.mode}/{image.size}"
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
        raise ValueError(f"Amplitude active bounds {bounds} exceed SLM {(width,height)}")
    canvas.paste(active, (left, top))
    return np.asarray(canvas, dtype=np.uint8), bounds


@torch.no_grad()
def _rank_test_samples(
    model: SingleLayerMNIST4D2NN,
    dataset: torch.utils.data.Dataset,
    settings: Settings,
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
        images = images.to(device, non_blocking=True)
        output = model(images)
        fractions = output["detector_fraction"].cpu()
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
    for label in settings.classes:
        candidates[label].sort(
            key=lambda item: (item["correct"], item["margin"]), reverse=True
        )
    return candidates


def export_hardware_bundle(
    model: SingleLayerMNIST4D2NN,
    test_dataset: torch.utils.data.Dataset,
    settings: Settings,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
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
    phase_canvas, phase_bounds, actual_phase_center = place_at_center(
        Image.fromarray(native_phase, mode="L"),
        slm_size_wh=settings.phase_slm_size_wh,
        center_xy=settings.phase_slm_center_xy,
    )
    phase_path = output_dir / "phase_to_play" / "mnist4_single_layer_17um_5cm.bmp"
    phase_sha = _save_checked_bmp(
        np.asarray(phase_canvas, dtype=np.uint8),
        phase_path,
        settings.phase_slm_size_wh,
    )

    ranking = _rank_test_samples(model, test_dataset, settings, device)
    sample_rows: list[dict[str, Any]] = []
    amplitude_bounds: tuple[int, int, int, int] | None = None
    for label in settings.classes:
        selected = ranking[label][: settings.export_samples_per_class]
        for class_rank, item in enumerate(selected):
            image, target = test_dataset[item["dataset_index"]]
            active = model.prepare_active_amplitude(image.unsqueeze(0)).squeeze(0)
            full, bounds = _full_amplitude_frame(active.numpy(), settings)
            amplitude_bounds = bounds
            key = f"mnist4_i{item['dataset_index']:05d}_y{int(target)}_r{class_rank:02d}"
            path = output_dir / "amplitude_to_play" / f"{key}.bmp"
            digest = _save_checked_bmp(full, path, settings.amplitude_slm_size_wh)
            sample_rows.append(
                {
                    "key": key,
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
                }
            )
    write_csv(output_dir / "samples.csv", sample_rows)

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
    contract = {
        "schema_version": 1,
        "model": "single-layer phase-only MNIST-4 D2NN",
        "classes": list(settings.classes),
        "wavelength_nm": settings.wavelength_nm,
        "phase_to_ccd_distance_cm": settings.detector_distance_m * 100.0,
        "input_phase_relation": "4F co-planar; no simulated free-space propagation before phase",
        "logical_geometry": {
            "canvas_size": settings.canvas_size,
            "active_size": settings.active_size,
            "input_size": settings.input_size,
            "detector_bounds_xyxy": [list(value) for value in settings.detector_bounds()],
        },
        "amplitude_slm": {
            "size_wh": list(settings.amplitude_slm_size_wh),
            "pixel_pitch_um": settings.amplitude_slm_pixel_pitch_um,
            "center_xy": list(settings.amplitude_slm_center_xy),
            "active_bounds_xyxy": list(amplitude_bounds or ()),
            "invert_before_export": settings.amplitude_invert_before_export,
            "outside_active_value_uint8": 255 if settings.amplitude_invert_before_export else 0,
            "bright_value_uint8": 0 if settings.amplitude_invert_before_export else 255,
            "dark_value_uint8": 255 if settings.amplitude_invert_before_export else 0,
        },
        "phase_slm": {
            "file": phase_path.name,
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
            "normalization": "detector energies divided by total frame energy; global gain invariant",
            "background_subtraction": False,
            "flip": "apply the experimentally measured Fresnel correspondence before classification",
        },
        "sample_count": len(sample_rows),
    }
    write_json(output_dir / "hardware_contract.json", contract)
    polarity_text = (
        "振幅BMP执行黑白反相：数字区域为低灰度、背景为高灰度。"
        if settings.amplitude_invert_before_export
        else "振幅BMP不反相：255=白/透光、0=黑/遮光，数字区域为高灰度、背景为0。"
    )
    (output_dir / "README.md").write_text(
        f"""# MNIST-4 hardware bundle

固定播放 `phase_to_play/mnist4_single_layer_17um_5cm.bmp`，再逐张播放
`amplitude_to_play/*.bmp` 并用相同文件主名保存CCD图像。{polarity_text}

CCD先使用四菲涅尔焦点标定得到的ROI裁剪，再按实验测得的上下/左右对应关系翻转，
最后以面积重采样到478×478。分类只对 `detector_regions.csv` 的四个区域积分并取最大值，
不做不存在的背景扣除；整帧乘性光强变化由能量比例自动消除。
""",
        encoding="utf-8",
    )
    return contract
