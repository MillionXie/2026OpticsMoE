import urllib.request
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .loader_utils import dataloader_kwargs


DSPRITES_FILENAME = "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
DSPRITES_URL = "https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
TASK_NUM_CLASSES = {
    "shape": 3,
    "scale": 6,
    "x_position_4bin": 4,
    "y_position_4bin": 4,
}


def dsprites_path(root: str) -> Path:
    return Path(root) / "dSprites" / DSPRITES_FILENAME


def ensure_dsprites_file(root: str, download: bool = True, npz_path: Optional[str] = None) -> Path:
    path = Path(npz_path) if npz_path else dsprites_path(root)
    if path.exists():
        return path
    if not download:
        raise FileNotFoundError(f"dSprites npz not found: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(DSPRITES_URL, str(path))
    return path


def normalize_task_names(tasks: Sequence[str]):
    names = [str(task).lower() for task in tasks]
    unsupported = [task for task in names if task not in TASK_NUM_CLASSES]
    if unsupported:
        raise ValueError(f"Unsupported dSprites multitask labels: {unsupported}")
    return names


def label_from_latents(latents_classes, task_name: str):
    if task_name == "shape":
        return latents_classes[:, 1].astype(np.int64)
    if task_name == "scale":
        return latents_classes[:, 2].astype(np.int64)
    if task_name == "x_position_4bin":
        return (latents_classes[:, 4] // 8).astype(np.int64)
    if task_name == "y_position_4bin":
        return (latents_classes[:, 5] // 8).astype(np.int64)
    raise KeyError(task_name)


def shuffled_split_indices(total_count: int, val_split: float, test_split: float, seed: int):
    rng = np.random.default_rng(int(seed))
    indices = np.arange(int(total_count), dtype=np.int64)
    rng.shuffle(indices)
    test_size = int(round(float(test_split) * total_count))
    val_size = int(round(float(val_split) * total_count))
    test_indices = indices[:test_size]
    val_indices = indices[test_size:test_size + val_size]
    train_indices = indices[test_size + val_size:]
    return {"train": train_indices, "val": val_indices, "test": test_indices}


def split_indices_from_config(total_count: int, dataset_cfg: Dict, val_split: float, test_split: float, seed: int):
    protocol = dataset_cfg.get("sampling_protocol", {}) or {}
    if not bool(protocol.get("enabled", False)):
        return shuffled_split_indices(total_count, val_split=val_split, test_split=test_split, seed=seed)
    total_requested = protocol.get("total_size")
    total_size = int(total_count) if total_requested is None else int(total_requested)
    if total_size <= 0 or total_size > int(total_count):
        raise ValueError(f"Invalid dSprites sampling total_size={total_size} for dataset length {total_count}.")
    ratio = protocol.get("train_test_ratio", [4, 1])
    train_parts, test_parts = int(ratio[0]), int(ratio[1])
    if train_parts <= 0 or test_parts <= 0:
        raise ValueError("sampling_protocol.train_test_ratio must contain positive values.")
    rng = np.random.default_rng(int(seed) + int(protocol.get("seed_offset", 0)))
    indices = np.arange(int(total_count), dtype=np.int64)
    rng.shuffle(indices)
    selected = indices[:total_size]
    train_pool_size = total_size * train_parts // (train_parts + test_parts)
    test_size = total_size - train_pool_size
    train_pool = selected[:train_pool_size]
    test_indices = selected[train_pool_size:train_pool_size + test_size]
    val_size = int(round(float(val_split) * len(train_pool)))
    val_indices = train_pool[:val_size]
    train_indices = train_pool[val_size:]
    return {"train": train_indices, "val": val_indices, "test": test_indices}


def apply_sampling_protocol(split_indices, dataset_cfg: Dict, seed: int):
    """Backward-compatible wrapper for older callers.

    New code should call split_indices_from_config so sampling_protocol.total_size
    is interpreted as train + val + test, not just train/test after a full split.
    """

    protocol = dataset_cfg.get("sampling_protocol", {}) or {}
    if not bool(protocol.get("enabled", False)):
        return split_indices
    total_available = sum(len(split_indices[key]) for key in ("train", "val", "test"))
    return split_indices_from_config(
        total_available,
        dataset_cfg,
        val_split=len(split_indices["val"]) / max(total_available, 1),
        test_split=len(split_indices["test"]) / max(total_available, 1),
        seed=seed,
    )


def apply_max_split_samples(split_indices, dataset_cfg: Dict):
    result = dict(split_indices)
    for split in ("train", "val", "test"):
        value = dataset_cfg.get(f"max_{split}_samples")
        if value is not None:
            result[split] = result[split][: int(value)]
    return result


class DSpritesSameInputMultiTaskDataset(Dataset):
    """dSprites dataset returning one image and labels for multiple tasks."""

    def __init__(
        self,
        root: str,
        tasks: Sequence[str],
        input_size: int = 134,
        split: str = "train",
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 7,
        download: bool = True,
        max_samples: Optional[int] = None,
        sampling_protocol: Optional[Dict] = None,
        npz_path: Optional[str] = None,
        indices=None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test.")
        self.tasks = normalize_task_names(tasks)
        self.input_size = int(input_size)
        self.split = split
        self.path = ensure_dsprites_file(root, download=download, npz_path=npz_path)
        self._npz = np.load(str(self.path), allow_pickle=False)
        self.imgs = self._npz["imgs"]
        self.latents_classes = self._npz["latents_classes"]
        if indices is None:
            splits = split_indices_from_config(
                len(self.imgs),
                {"sampling_protocol": sampling_protocol or {}},
                val_split=val_split,
                test_split=test_split,
                seed=seed,
            )
            indices = splits[split]
        self.indices = np.asarray(indices, dtype=np.int64)
        if max_samples is not None:
            self.indices = self.indices[: int(max_samples)]

    def __len__(self) -> int:
        return int(len(self.indices))

    def _image_tensor(self, base_index: int) -> torch.Tensor:
        image = torch.from_numpy(np.asarray(self.imgs[int(base_index)], dtype=np.float32)).unsqueeze(0)
        if tuple(image.shape[-2:]) != (self.input_size, self.input_size):
            image = F.interpolate(image.unsqueeze(0), size=(self.input_size, self.input_size), mode="bilinear", align_corners=False).squeeze(0)
        return image.clamp(0.0, 1.0)

    def __getitem__(self, index: int):
        base_index = int(self.indices[int(index)])
        latent_row = self.latents_classes[base_index:base_index + 1]
        labels = {
            task: torch.tensor(int(label_from_latents(latent_row, task)[0]), dtype=torch.long)
            for task in self.tasks
        }
        return self._image_tensor(base_index), labels


def create_same_input_multitask_dataloaders(config: Dict, seed: int = 7):
    dataset_cfg = config.get("dataset", {})
    training_cfg = config.get("training", {})
    task_names = normalize_task_names(training_cfg.get("tasks", ["shape", "scale"]))
    root = dataset_cfg.get("root", "./data")
    input_size = int(dataset_cfg.get("input_size", 134))
    val_split = float(dataset_cfg.get("val_split", 0.1))
    test_split = float(dataset_cfg.get("test_split", 0.1))
    ds_seed = int(dataset_cfg.get("seed", seed))
    download = bool(dataset_cfg.get("download", True))
    npz_path = dataset_cfg.get("npz_path")
    path = ensure_dsprites_file(root, download=download, npz_path=npz_path)
    with np.load(str(path), allow_pickle=False) as npz:
        total_count = int(npz["imgs"].shape[0])
    splits = split_indices_from_config(total_count, dataset_cfg, val_split=val_split, test_split=test_split, seed=ds_seed)
    if bool(dataset_cfg.get("smoke_test", False)):
        splits["train"] = splits["train"][: int(dataset_cfg.get("smoke_train_size", 64))]
        splits["val"] = splits["val"][: int(dataset_cfg.get("smoke_test_size", 32))]
        splits["test"] = splits["test"][: int(dataset_cfg.get("smoke_test_size", 32))]
    splits = apply_max_split_samples(splits, dataset_cfg)
    common = {
        "root": root,
        "tasks": task_names,
        "input_size": input_size,
        "val_split": val_split,
        "test_split": test_split,
        "seed": ds_seed,
        "download": download,
        "sampling_protocol": dataset_cfg.get("sampling_protocol", {}),
        "npz_path": str(path),
    }
    train = DSpritesSameInputMultiTaskDataset(split="train", indices=splits["train"], **common)
    val = DSpritesSameInputMultiTaskDataset(split="val", indices=splits["val"], **common)
    test = DSpritesSameInputMultiTaskDataset(split="test", indices=splits["test"], **common)
    return (
        DataLoader(train, **dataloader_kwargs(dataset_cfg, shuffle=True, seed=ds_seed + 300_000)),
        DataLoader(val, **dataloader_kwargs(dataset_cfg, shuffle=False)),
        DataLoader(test, **dataloader_kwargs(dataset_cfg, shuffle=False)),
        {task: TASK_NUM_CLASSES[task] for task in task_names},
        task_names,
    )
