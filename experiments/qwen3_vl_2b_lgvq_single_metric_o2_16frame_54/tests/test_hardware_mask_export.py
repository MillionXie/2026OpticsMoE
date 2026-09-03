from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from .. import export_hardware_masks as exporter
from ..export_hardware_masks import STAGES, build_stage_planes, export_hardware_masks
from ..modeling import build_model
from ..settings import load_settings


def _settings():
    config = (
        Path(__file__).parents[1]
        / "configs"
        / "release"
        / "temporal.yaml"
    )
    return load_settings(config)


def test_six_phase_planes_use_the_exact_training_coordinates() -> None:
    settings = _settings()
    model = build_model(settings)
    planes = build_stage_planes(model, settings)

    assert tuple(planes) == STAGES
    assert all(plane.phase_rad.shape == (478, 478) for plane in planes.values())
    assert len(planes["vision_router"].tile_phase_rad) == 16
    assert len(planes["vision_expert"].tile_phase_rad) == 64
    assert len(planes["vision_global"].tile_phase_rad) == 1
    assert len(planes["language_router"].tile_phase_rad) == 1
    assert len(planes["language_expert"].tile_phase_rad) == 4
    assert len(planes["language_global"].tile_phase_rad) == 1

    # 54x54 router tiles are centered in each 114x114 lane.
    assert planes["vision_router"].learned_boxes_xyxy[0] == (32, 32, 86, 86)
    assert planes["vision_router"].learned_boxes_xyxy[-1] == (392, 392, 446, 446)
    # 64 experts exactly reproduce the per-lane 2x2 training placement.
    assert planes["vision_expert"].learned_boxes_xyxy[:4] == (
        (2, 2, 56, 56),
        (62, 2, 116, 56),
        (2, 62, 56, 116),
        (62, 62, 116, 116),
    )
    assert planes["vision_expert"].learned_boxes_xyxy[-1] == (422, 422, 476, 476)
    # 109 is odd: the exact integer placement used by modeling.py starts at 184.
    assert planes["language_router"].learned_boxes_xyxy == ((184, 184, 293, 293),)
    assert planes["language_expert"].learned_boxes_xyxy == (
        (123, 123, 232, 232),
        (246, 123, 355, 232),
        (123, 246, 232, 355),
        (246, 246, 355, 355),
    )
    assert bool(planes["vision_global"].learned_support.all())
    assert bool(planes["language_global"].learned_support.all())


def test_export_writes_native_bmps_and_preserves_canonical_orientation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings()
    model = build_model(settings)
    checkpoint = tmp_path / "temporal.pt"
    torch.save(
        {
            "architecture": settings.architecture_label,
            "target_name": settings.target_name,
            "prompt": settings.prompt,
            "epoch": 7,
            "state_dict": model.state_dict(),
        },
        checkpoint,
    )
    # Preview rendering is orthogonal to this raster/geometry contract and is
    # skipped here to keep the unit test fast and backend-independent.
    monkeypatch.setattr(exporter, "_save_previews", lambda *_args, **_kwargs: None)
    report = export_hardware_masks(settings, checkpoint, tmp_path / "export")

    assert report["target_name"] == "temporal"
    assert report["checkpoint_epoch"] == 7
    assert report["phase_slm"]["native_active_size_wh"] == [1016, 1016]
    assert report["phase_slm"]["extent_error_um_vs_logical"] == pytest.approx(2.0)
    assert report["phase_slm"]["reconstruction"]["active_center_xy"] == [980.0, 590.0]
    assert report["phase_slm"]["reconstruction"]["physical_ratio"] == 2.125
    assert report["amplitude_slm"]["active_bounds_xyxy"] == [273, 273, 751, 751]

    canonical_path = (
        tmp_path / "export" / "logical_phase_478_canonical" / "vision_router.png"
    )
    payload_path = (
        tmp_path
        / "export"
        / "phase_payload_478_hardware_orientation"
        / "vision_router.png"
    )
    with Image.open(canonical_path) as image:
        canonical = np.asarray(image)
        assert image.mode == "L" and image.size == (478, 478)
    with Image.open(payload_path) as image:
        payload = np.asarray(image)
    assert np.array_equal(payload, np.flip(canonical, axis=0))

    for stage in STAGES:
        with Image.open(
            tmp_path / "export" / "phase_slm_1920x1200" / f"{stage}.bmp"
        ) as image:
            assert image.mode == "L"
            assert image.size == (1920, 1200)
        with Image.open(
            tmp_path
            / "export"
            / "amplitude_layout_1024x1024"
            / f"{stage}_layout_1024x1024.bmp"
        ) as image:
            assert image.mode == "L"
            assert image.size == (1024, 1024)

    persisted = json.loads(
        (tmp_path / "export" / "hardware_mask_export_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert persisted["stages_in_physical_order"] == list(STAGES)
    assert persisted["phase_slm"]["flip_vertical_before_physical_raster"] is True


def test_spatial_checkpoint_cannot_be_exported_with_temporal_config(tmp_path: Path) -> None:
    spatial = load_settings(
        Path(__file__).parents[1] / "configs" / "release" / "spatial.yaml"
    )
    model = build_model(spatial)
    checkpoint = tmp_path / "spatial.pt"
    torch.save(
        {
            "architecture": spatial.architecture_label,
            "target_name": "spatial",
            "prompt": spatial.prompt,
            "state_dict": model.state_dict(),
        },
        checkpoint,
    )
    temporal = load_settings(
        Path(__file__).parents[1] / "configs" / "release" / "temporal.yaml"
    )
    with pytest.raises(RuntimeError, match="architecture mismatch"):
        exporter._load_checkpoint_model(temporal, checkpoint)
