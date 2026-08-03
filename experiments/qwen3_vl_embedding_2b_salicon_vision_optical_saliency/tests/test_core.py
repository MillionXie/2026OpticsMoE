from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.datasets import (
    SALICONSaliencyDataset,
    _find_image_directory,
    _iter_json_array,
    prepare_salicon,
)
from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.modeling import (
    ContinuousSaliencyHead,
    restore_detector_spatial,
    restore_packed_spatial,
)
from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.objectives import (
    SaliencyAccumulator,
    auc_judd,
    density_from_logits,
    saliency_loss,
)
from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.settings import (
    load_settings,
)


EXPERIMENT = Path(__file__).resolve().parents[1]


def test_formal_config_keeps_one_layer_moe16_geometry() -> None:
    settings = load_settings(EXPERIMENT / "configs" / "salicon.yaml")
    assert settings.image_size == 224
    assert settings.processor_min_pixels == 50176
    assert settings.output_dir == (EXPERIMENT / "runs/salicon_vision_optical_saliency").resolve()
    assert (
        settings.canvas_size,
        settings.active_size,
        settings.expert_size,
        settings.expert_pitch,
        settings.num_experts,
        settings.top_k,
        settings.expert_layers,
        settings.detector_output_size,
    ) == (1026, 986, 224, 254, 16, 4, 1, 224)


def test_mask_kd_config_disables_unaligned_augmentation() -> None:
    settings = load_settings(EXPERIMENT / "configs" / "salicon_mask_kd.yaml")
    assert settings.map_kd_weight > 0
    assert settings.augmentation_enabled is False
    assert settings.teacher_checkpoint is not None


def test_streaming_json_array_reader(tmp_path: Path) -> None:
    path = tmp_path / "annotation.json"
    payload = {
        "images": [{"id": 2}, {"id": 1}],
        "annotations": [
            {"image_id": 1, "fixations": [[1, 2], [3, 4]]},
            {"image_id": 2, "fixations": []},
        ],
    }
    try:
        path.write_text(json.dumps(payload), encoding="utf-8")
        assert [row["id"] for row in _iter_json_array(path, "images")] == [2, 1]
        rows = list(_iter_json_array(path, "annotations"))
        assert rows[0]["fixations"] == [[1, 2], [3, 4]]
    finally:
        path.unlink(missing_ok=True)


def test_prepare_and_load_synthetic_salicon_maps(tmp_path: Path) -> None:
    for folder in ("images/train2014", "images/val2014", "annotations"):
        (tmp_path / folder).mkdir(parents=True, exist_ok=True)
    rows = {}
    for split, folder, image_id in (
        ("train", "train2014", 11),
        ("validation", "val2014", 22),
    ):
        filename = f"COCO_{folder}_{image_id:012d}.jpg"
        Image.new("RGB", (32, 24), (100, 120, 140)).save(
            tmp_path / "images" / folder / filename
        )
        rows[split] = {
            "images": [
                {"id": image_id, "file_name": filename, "width": 32, "height": 24}
            ],
            "annotations": [
                {
                    "image_id": image_id,
                    "worker_id": 1,
                    "fixations": [[4, 5], [12, 16], [20, 28]],
                }
            ],
        }
    (tmp_path / "annotations" / "fixations_train2014.json").write_text(
        json.dumps(rows["train"]), encoding="utf-8"
    )
    (tmp_path / "annotations" / "fixations_val2014.json").write_text(
        json.dumps(rows["validation"]), encoding="utf-8"
    )
    settings = SimpleNamespace(
        data_root=tmp_path,
        output_dir=tmp_path / "run",
        artifact_cache_dir=tmp_path / "cache",
        download=False,
        train_limit=None,
        validation_limit=None,
        materialize_density_maps=True,
        image_size=224,
        density_sigma_px=3.0,
        augmentation_enabled=False,
        crop_scale_min=0.9,
        horizontal_flip_probability=0.5,
        brightness_jitter=0.0,
        contrast_jitter=0.0,
    )
    bundle = prepare_salicon(settings)
    assert len(bundle.train_records) == len(bundle.validation_records) == 1
    assert bundle.train_records[0].image_id != bundle.validation_records[0].image_id
    item = SALICONSaliencyDataset(
        bundle.train_records, settings, training=False
    )[0]
    assert item["density"].shape == item["fixation"].shape == (1, 224, 224)
    assert float(item["density"].sum()) == pytest.approx(1.0, abs=1e-5)
    assert set(torch.unique(item["fixation"]).tolist()) <= {0.0, 1.0}


def test_official_salicon_train_val_archive_directory_aliases(
    tmp_path: Path,
) -> None:
    train = tmp_path / "images" / "train"
    validation = tmp_path / "images" / "val"
    train.mkdir(parents=True)
    validation.mkdir(parents=True)
    Image.new("RGB", (8, 8), "white").save(
        train / "COCO_train2014_000000000001.jpg"
    )
    Image.new("RGB", (8, 8), "white").save(
        validation / "COCO_val2014_000000000002.jpg"
    )
    assert _find_image_directory(tmp_path, "train") == train
    assert _find_image_directory(tmp_path, "validation") == validation


def test_saliency_objective_is_finite_and_differentiable() -> None:
    target = torch.zeros(2, 1, 16, 16)
    fixation = torch.zeros_like(target)
    target[:, :, 6:10, 7:11] = 1.0
    fixation[:, :, 8, 9] = 1.0
    logits = torch.randn_like(target, requires_grad=True)
    settings = SimpleNamespace(
        kl_weight=1.0,
        cc_weight=0.5,
        sim_weight=0.25,
        nss_weight=0.1,
        map_kd_weight=0.0,
        map_kd_temperature=1.0,
    )
    loss, pieces = saliency_loss(logits, target, fixation, settings)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert set(pieces) == {"kl", "cc", "sim", "nss", "map_kd"}
    assert torch.allclose(
        density_from_logits(logits).flatten(1).sum(dim=1),
        torch.ones(2),
    )


def test_perfect_density_metrics_and_fast_auc() -> None:
    fixation = torch.zeros(1, 1, 8, 8)
    fixation[:, :, 2:4, 5:7] = 1.0
    logits = torch.full_like(fixation, -10.0)
    logits[fixation.bool()] = 10.0
    prediction = density_from_logits(logits)
    assert auc_judd(prediction[0], fixation[0]) > 0.99
    assert auc_judd(torch.ones_like(prediction[0]), fixation[0]) == pytest.approx(
        0.5
    )
    accumulator = SaliencyAccumulator()
    accumulator.update(logits, prediction, fixation)
    metrics = accumulator.compute()
    assert metrics["kld"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["cc"] == pytest.approx(1.0, abs=1e-5)
    assert metrics["sim"] == pytest.approx(1.0, abs=1e-5)


def test_continuous_head_preserves_spatial_output_and_gradients() -> None:
    head = ContinuousSaliencyHead(224, 64, (32, 16), 8, output_size=224)
    features = torch.randn(2, 224, 14, 14, requires_grad=True)
    logits = head(features)
    assert logits.shape == (2, 1, 224, 224)
    logits.mean().backward()
    assert features.grad is not None
    assert sum(parameter.numel() for parameter in head.parameters()) < 250_000


def test_runtime_token_grid_restoration_is_strict() -> None:
    grid = torch.tensor([[1, 2, 3], [1, 2, 3]])
    packed = torch.randn(12, 7)
    assert restore_packed_spatial(packed, grid).shape == (2, 7, 2, 3)
    detector = torch.randn(2, 224, 224)
    assert restore_detector_spatial(detector, grid, [6, 6]).shape == (
        2,
        224,
        2,
        3,
    )
    with pytest.raises(RuntimeError, match="does not match"):
        restore_packed_spatial(torch.randn(11, 7), grid)
