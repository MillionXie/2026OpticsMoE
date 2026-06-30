import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.data.dsprites_multitask import DSpritesSameInputMultiTaskDataset


def _fake_npz(path: Path, count: int = 96):
    rng = np.random.default_rng(0)
    imgs = rng.integers(0, 2, size=(count, 64, 64), dtype=np.uint8)
    latents = np.zeros((count, 6), dtype=np.int64)
    latents[:, 1] = np.arange(count) % 3
    latents[:, 2] = np.arange(count) % 6
    latents[:, 4] = np.arange(count) % 32
    latents[:, 5] = (np.arange(count) * 3) % 32
    np.savez(path, imgs=imgs, latents_classes=latents, latents_values=latents.astype(np.float32))


def test_dsprites_same_input_labels_and_splits(tmp_path):
    npz_path = tmp_path / "dsprites.npz"
    _fake_npz(npz_path)
    tasks = ["shape", "scale", "x_position_4bin", "y_position_4bin"]
    train = DSpritesSameInputMultiTaskDataset(tmp_path, tasks, input_size=120, split="train", download=False, npz_path=str(npz_path), seed=7)
    val = DSpritesSameInputMultiTaskDataset(tmp_path, tasks, input_size=120, split="val", download=False, npz_path=str(npz_path), seed=7)
    test = DSpritesSameInputMultiTaskDataset(tmp_path, tasks, input_size=120, split="test", download=False, npz_path=str(npz_path), seed=7)
    assert set(train.indices).isdisjoint(set(val.indices))
    assert set(train.indices).isdisjoint(set(test.indices))
    image, labels = train[0]
    assert tuple(image.shape) == (1, 120, 120)
    assert 0 <= int(labels["shape"]) <= 2
    assert 0 <= int(labels["scale"]) <= 5
    assert 0 <= int(labels["x_position_4bin"]) <= 3
    assert 0 <= int(labels["y_position_4bin"]) <= 3
    image2, labels2 = train[0]
    assert image.equal(image2)
    assert {key: int(value) for key, value in labels.items()} == {key: int(value) for key, value in labels2.items()}
