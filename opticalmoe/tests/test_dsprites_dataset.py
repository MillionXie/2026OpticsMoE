from pathlib import Path

import numpy as np
import pytest
import torch

from opticalmoe.data.dsprites import (
    DSpritesMultiLabelDataset,
    DSpritesTaskDataset,
    create_dsprites_dataloaders,
)


def _write_fake_dsprites(path: Path, n: int = 120):
    rng = np.random.default_rng(0)
    imgs = rng.integers(0, 2, size=(n, 64, 64), dtype=np.uint8)
    latents_classes = np.zeros((n, 6), dtype=np.int64)
    latents_classes[:, 1] = np.arange(n) % 3
    latents_classes[:, 2] = np.arange(n) % 6
    latents_values = latents_classes.astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, imgs=imgs, latents_classes=latents_classes, latents_values=latents_values)
    return path


def test_dsprites_missing_file_download_false(tmp_path):
    with pytest.raises(FileNotFoundError, match="dSprites npz not found"):
        DSpritesTaskDataset(
            root=str(tmp_path),
            task_name="shape",
            download=False,
        )


def test_dsprites_shape_and_scale_labels_and_image_shape(tmp_path):
    npz_path = _write_fake_dsprites(tmp_path / "fake_dsprites.npz")
    shape_ds = DSpritesTaskDataset(
        root=str(tmp_path),
        task_name="shape",
        input_size=134,
        split="train",
        download=False,
        npz_path=str(npz_path),
    )
    scale_ds = DSpritesTaskDataset(
        root=str(tmp_path),
        task_name="scale",
        input_size=134,
        split="train",
        download=False,
        npz_path=str(npz_path),
    )

    image, shape_label = shape_ds[0]
    _image2, scale_label = scale_ds[0]
    assert image.shape == (1, 134, 134)
    assert image.dtype == torch.float32
    assert 0 <= shape_label <= 2
    assert 0 <= scale_label <= 5


def test_dsprites_train_val_test_split_non_overlapping(tmp_path):
    npz_path = _write_fake_dsprites(tmp_path / "fake_dsprites.npz")
    train = DSpritesTaskDataset(str(tmp_path), "shape", split="train", download=False, npz_path=str(npz_path))
    val = DSpritesTaskDataset(str(tmp_path), "shape", split="val", download=False, npz_path=str(npz_path))
    test = DSpritesTaskDataset(str(tmp_path), "shape", split="test", download=False, npz_path=str(npz_path))

    assert set(train.indices).isdisjoint(set(val.indices))
    assert set(train.indices).isdisjoint(set(test.indices))
    assert set(val.indices).isdisjoint(set(test.indices))


def test_dsprites_multilabel_matches_latent_columns(tmp_path):
    npz_path = _write_fake_dsprites(tmp_path / "fake_dsprites.npz")
    ds = DSpritesMultiLabelDataset(
        root=str(tmp_path),
        input_size=134,
        split="test",
        download=False,
        npz_path=str(npz_path),
    )
    image, labels = ds[0]
    base_index = int(ds.indices[0])
    with np.load(str(npz_path), allow_pickle=False) as npz:
        expected_shape = int(npz["latents_classes"][base_index, 1])
        expected_scale = int(npz["latents_classes"][base_index, 2])
    assert image.shape == (1, 134, 134)
    assert labels["shape"] == expected_shape
    assert labels["scale"] == expected_scale


def test_create_dsprites_dataloaders_class_counts_and_aligned_indices(tmp_path):
    npz_path = _write_fake_dsprites(tmp_path / "fake_dsprites.npz", n=180)
    cfg = {
        "name": "dsprites",
        "task": "shape",
        "root": str(tmp_path),
        "npz_path": str(npz_path),
        "input_size": 134,
        "batch_size": 4,
        "download": False,
        "seed": 7,
        "sampling_protocol": {"enabled": True, "total_size": 60, "train_test_ratio": [4, 1]},
    }
    shape_train, _shape_val, _shape_test, shape_classes = create_dsprites_dataloaders(cfg, seed=123)
    cfg["task"] = "scale"
    scale_train, _scale_val, _scale_test, scale_classes = create_dsprites_dataloaders(cfg, seed=999)
    assert shape_classes == 3
    assert scale_classes == 6
    assert np.array_equal(shape_train.dataset.indices, scale_train.dataset.indices)
