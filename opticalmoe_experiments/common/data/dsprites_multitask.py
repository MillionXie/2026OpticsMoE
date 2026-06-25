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


def apply_sampling_protocol(split_indices, dataset_cfg: Dict, seed: int):
    protocol = dataset_cfg.get("sampling_protocol", {}) or {}
    if not bool(protocol.get("enabled", False)):
        return split_indices
    total_size = min(int(protocol.get("total_size", len(split_indices["train"]) + len(split_indices["test"]))), len(split_indices["train"]) + len(split_indices["test"]))
    ratio = protocol.get("train_test_ratio", [4, 1])
    train_parts, test_parts = int(ratio[0]), int(ratio[1])
    train_size = total_size * train_parts // max(train_parts + test_parts, 1)
    test_size = total_size - train_size
    rng = np.random.default_rng(int(seed) + int(protocol.get("seed_offset", 0)))
    train_pool = np.array(split_indices["train"], copy=True)
    test_pool = np.array(split_indices["test"], copy=True)
    rng.shuffle(train_pool)
    rng.shuffle(test_pool)
    split_indices = dict(split_indices)
    split_indices["train"] = train_pool[:train_size]
    split_indices["test"] = test_pool[:test_size]
    return split_indices


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
            splits = shuffled_split_indices(len(self.imgs), val_split=val_split, test_split=test_split, seed=seed)
            splits = apply_sampling_protocol(splits, {"sampling_protocol": sampling_protocol or {}}, seed)
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
    splits = shuffled_split_indices(total_count, val_split=val_split, test_split=test_split, seed=ds_seed)
    splits = apply_sampling_protocol(splits, dataset_cfg, ds_seed)
    if bool(dataset_cfg.get("smoke_test", False)):
        splits["train"] = splits["train"][: int(dataset_cfg.get("smoke_train_size", 64))]
        splits["val"] = splits["val"][: int(dataset_cfg.get("smoke_test_size", 32))]
        splits["test"] = splits["test"][: int(dataset_cfg.get("smoke_test_size", 32))]
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
