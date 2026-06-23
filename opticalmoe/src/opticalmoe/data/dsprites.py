import urllib.request
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


DSPRITES_FILENAME = "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
DSPRITES_URL = (
    "https://github.com/deepmind/dsprites-dataset/raw/master/"
    "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
)
DSPRITES_LABEL_COLUMNS = {"shape": 1, "scale": 2}
DSPRITES_NUM_CLASSES = {"shape": 3, "scale": 6}


def dsprites_path(root: str) -> Path:
    return Path(root) / "dSprites" / DSPRITES_FILENAME


def ensure_dsprites_file(root: str, download: bool = True, npz_path: Optional[str] = None) -> Path:
    path = Path(npz_path) if npz_path else dsprites_path(root)
    if path.exists():
        return path
    if not download:
        raise FileNotFoundError(
            f"dSprites npz not found: {path}. Put {DSPRITES_FILENAME} there "
            "or set download=true."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(DSPRITES_URL, str(path))
    if not path.exists():
        raise RuntimeError(f"Download did not create dSprites file: {path}")
    return path


def shuffled_split_indices(total_count: int, val_split: float, test_split: float, seed: int):
    rng = np.random.default_rng(int(seed))
    indices = np.arange(int(total_count), dtype=np.int64)
    rng.shuffle(indices)
    test_size = int(round(float(test_split) * total_count))
    val_size = int(round(float(val_split) * total_count))
    test_indices = indices[:test_size]
    val_indices = indices[test_size : test_size + val_size]
    train_indices = indices[test_size + val_size :]
    return {"train": train_indices, "val": val_indices, "test": test_indices}


def apply_sampling_protocol(indices, dataset_cfg: Dict, seed: int):
    protocol = dataset_cfg.get("sampling_protocol", {})
    if not protocol or not bool(protocol.get("enabled", False)):
        return indices, None
    total_size = min(int(protocol["total_size"]), len(indices))
    ratio = protocol.get("train_test_ratio", [4, 1])
    if len(ratio) != 2:
        raise ValueError("sampling_protocol.train_test_ratio must have two values.")
    train_parts = int(ratio[0])
    test_parts = int(ratio[1])
    if train_parts <= 0 or test_parts <= 0:
        raise ValueError("train_test_ratio values must be positive.")
    train_pool_size = total_size * train_parts // (train_parts + test_parts)
    test_size = total_size - train_pool_size
    rng = np.random.default_rng(int(seed) + int(protocol.get("seed_offset", 0)))
    selected = np.array(indices, copy=True)
    rng.shuffle(selected)
    return selected[:train_pool_size], selected[train_pool_size : train_pool_size + test_size]


class DSpritesTaskDataset(Dataset):
    """Single dSprites task dataset: shape or scale classification."""

    def __init__(
        self,
        root: str,
        task_name: str,
        input_size: int = 134,
        split: str = "train",
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 7,
        download: bool = True,
        max_samples: Optional[int] = None,
        indices=None,
        npz_path: Optional[str] = None,
    ) -> None:
        task_name = str(task_name).lower()
        if task_name not in DSPRITES_LABEL_COLUMNS:
            raise ValueError("task_name must be 'shape' or 'scale'.")
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test.")
        self.root = root
        self.task_name = task_name
        self.input_size = int(input_size)
        self.split = split
        self.path = ensure_dsprites_file(root, download=download, npz_path=npz_path)
        self._npz = np.load(str(self.path), allow_pickle=False)
        self.imgs = self._npz["imgs"]
        self.latents_classes = self._npz["latents_classes"]
        if indices is None:
            splits = shuffled_split_indices(
                self.latents_classes.shape[0],
                val_split=val_split,
                test_split=test_split,
                seed=seed,
            )
            indices = splits[split]
        self.indices = np.asarray(indices, dtype=np.int64)
        if max_samples is not None:
            self.indices = self.indices[: int(max_samples)]

    @property
    def labels(self) -> torch.Tensor:
        col = DSPRITES_LABEL_COLUMNS[self.task_name]
        return torch.as_tensor(self.latents_classes[self.indices, col], dtype=torch.long)

    @property
    def targets(self) -> torch.Tensor:
        return self.labels

    def __len__(self) -> int:
        return int(len(self.indices))

    def image_tensor(self, base_index: int) -> torch.Tensor:
        image = torch.from_numpy(np.asarray(self.imgs[int(base_index)], dtype=np.float32)).unsqueeze(0)
        if tuple(image.shape[-2:]) != (self.input_size, self.input_size):
            image = F.interpolate(
                image.unsqueeze(0),
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return image.clamp(0.0, 1.0)

    def __getitem__(self, index: int):
        base_index = int(self.indices[int(index)])
        col = DSPRITES_LABEL_COLUMNS[self.task_name]
        return self.image_tensor(base_index), int(self.latents_classes[base_index, col])


class DSpritesMultiLabelDataset(Dataset):
    """dSprites dataset returning both shape and scale labels for one image."""

    def __init__(
        self,
        root: str,
        input_size: int = 134,
        split: str = "test",
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 7,
        download: bool = True,
        max_samples: Optional[int] = None,
        indices=None,
        npz_path: Optional[str] = None,
    ) -> None:
        self.base = DSpritesTaskDataset(
            root=root,
            task_name="shape",
            input_size=input_size,
            split=split,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
            download=download,
            max_samples=max_samples,
            indices=indices,
            npz_path=npz_path,
        )

    @property
    def indices(self):
        return self.base.indices

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        base_index = int(self.base.indices[int(index)])
        labels = {
            "shape": int(self.base.latents_classes[base_index, DSPRITES_LABEL_COLUMNS["shape"]]),
            "scale": int(self.base.latents_classes[base_index, DSPRITES_LABEL_COLUMNS["scale"]]),
        }
        return self.base.image_tensor(base_index), labels


def create_dsprites_dataloaders(dataset_cfg: Dict, seed: int):
    seed = int(dataset_cfg.get("seed", seed))
    task_name = dataset_cfg.get("task", dataset_cfg.get("task_name", "shape")).lower()
    if task_name not in DSPRITES_NUM_CLASSES:
        raise ValueError("dSprites task must be shape or scale.")
    root = dataset_cfg.get("root", "./data")
    input_size = int(dataset_cfg.get("input_size", 134))
    val_split = float(dataset_cfg.get("val_split", 0.1))
    test_split = float(dataset_cfg.get("test_split", 0.1))
    batch_size = int(dataset_cfg.get("batch_size", 8))
    num_workers = int(dataset_cfg.get("num_workers", 0))
    download = bool(dataset_cfg.get("download", True))
    smoke_test = bool(dataset_cfg.get("smoke_test", False))
    npz_path = dataset_cfg.get("npz_path")

    path = ensure_dsprites_file(root, download=download, npz_path=npz_path)
    with np.load(str(path), allow_pickle=False) as npz:
        total_count = int(npz["latents_classes"].shape[0])
    splits = shuffled_split_indices(total_count, val_split, test_split, seed)
    train_pool, sampled_test = apply_sampling_protocol(splits["train"], dataset_cfg, seed)
    test_indices = sampled_test if sampled_test is not None else splits["test"]
    val_size = max(1, int(round(len(train_pool) * val_split)))
    val_indices = train_pool[:val_size]
    train_indices = train_pool[val_size:]
    if smoke_test:
        train_indices = train_indices[: int(dataset_cfg.get("smoke_train_size", 256))]
        test_indices = test_indices[: int(dataset_cfg.get("smoke_test_size", 128))]
        val_indices = val_indices[: max(1, min(len(val_indices), int(dataset_cfg.get("smoke_test_size", 128))))]

    common = {
        "root": root,
        "task_name": task_name,
        "input_size": input_size,
        "val_split": val_split,
        "test_split": test_split,
        "seed": seed,
        "download": download,
        "npz_path": str(path),
    }
    train = DSpritesTaskDataset(split="train", indices=train_indices, **common)
    val = DSpritesTaskDataset(split="val", indices=val_indices, **common)
    test = DSpritesTaskDataset(split="test", indices=test_indices, **common)
    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(train, shuffle=True, **kwargs),
        DataLoader(val, shuffle=False, **kwargs),
        DataLoader(test, shuffle=False, **kwargs),
        DSPRITES_NUM_CLASSES[task_name],
    )
