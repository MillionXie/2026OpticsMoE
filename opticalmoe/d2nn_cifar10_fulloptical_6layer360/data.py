import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import datasets, transforms

from utils import resolve_path


class PILToFloatTensor:
    def __call__(self, image):
        image = image.convert("L")
        width, height = image.size
        value = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).view(height, width).unsqueeze(0)
        return value.float().div_(255.0)


class RemappedCIFARSubset(Dataset):
    def __init__(self, dataset, class_indices, samples_per_class=None, seed=7):
        self.dataset = dataset
        self.class_indices = [int(value) for value in class_indices]
        self.label_map = {value: index for index, value in enumerate(self.class_indices)}
        targets = torch.as_tensor(dataset.targets)
        generator = torch.Generator().manual_seed(int(seed))
        selected = []
        for source_class in self.class_indices:
            indices = (targets == source_class).nonzero().flatten()
            if samples_per_class is not None:
                order = torch.randperm(len(indices), generator=generator)
                indices = indices[order[: min(int(samples_per_class), len(indices))]]
            selected.extend(indices.tolist())
        order = torch.randperm(len(selected), generator=generator).tolist()
        self.indices = [selected[index] for index in order]
        selected_targets = targets[self.indices]
        self.targets = [self.label_map[int(value)] for value in selected_targets.tolist()]
        self.class_counts = {
            self.label_map[source_class]: int((selected_targets == source_class).sum())
            for source_class in self.class_indices
        }

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, label = self.dataset[self.indices[index]]
        return image, self.label_map[int(label)]


class PerClassEpochSampler(Sampler):
    """Balanced rotating per-epoch subset without shrinking the dataset."""

    def __init__(self, dataset, samples_per_class, seed=7):
        self.samples_per_class = int(samples_per_class)
        if self.samples_per_class <= 0:
            raise ValueError("train_samples_per_class_per_epoch must be positive or null")
        self.seed = int(seed); self.epoch = 0
        labels = torch.as_tensor(dataset.targets)
        self.groups = {int(label): (labels == label).nonzero().flatten() for label in labels.unique(sorted=True)}

    def __len__(self):
        return sum(min(self.samples_per_class, len(group)) for group in self.groups.values())

    def _permutation(self, label, cycle):
        generator = torch.Generator().manual_seed(self.seed + int(label) * 1009 + int(cycle) * 100003)
        group = self.groups[label]
        return group[torch.randperm(len(group), generator=generator)]

    def __iter__(self):
        selected = []
        for label, group in self.groups.items():
            count = min(self.samples_per_class, len(group)); offset = self.epoch * count
            cycle, start = divmod(offset, len(group)); remaining = count
            while remaining:
                permutation = self._permutation(label, cycle); take = min(remaining, len(group) - start)
                selected.extend(permutation[start:start + take].tolist())
                remaining -= take; cycle += 1; start = 0
        generator = torch.Generator().manual_seed(self.seed + self.epoch * 104729 + 17)
        order = torch.randperm(len(selected), generator=generator).tolist(); self.epoch += 1
        return iter([selected[index] for index in order])


def create_loaders(config, seed=7, smoke_test=False):
    cfg = config.get("dataset", {})
    resize_size = int(cfg.get("resize_size", 300))
    interpolation_name = str(cfg.get("interpolation", "bicubic")).lower()
    modes = {
        "nearest": transforms.InterpolationMode.NEAREST,
        "bilinear": transforms.InterpolationMode.BILINEAR,
        "bicubic": transforms.InterpolationMode.BICUBIC,
    }
    if interpolation_name not in modes:
        raise ValueError("dataset.interpolation must be nearest, bilinear, or bicubic")
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(
            (resize_size, resize_size),
            interpolation=modes[interpolation_name],
            antialias=bool(cfg.get("antialias", True)),
        ),
        PILToFloatTensor(),
    ])
    root = resolve_path(cfg.get("root", "./data"))
    base_train = datasets.CIFAR10(str(root), train=True, download=bool(cfg.get("download", True)), transform=transform)
    base_test = datasets.CIFAR10(str(root), train=False, download=bool(cfg.get("download", True)), transform=transform)
    classes = [int(value) for value in cfg.get("class_indices", [0, 1, 2, 3])]
    train_limit = cfg.get("train_samples_per_class")
    test_limit = cfg.get("test_samples_per_class")
    if smoke_test:
        train_limit = int(cfg.get("smoke_train_per_class", 4))
        test_limit = int(cfg.get("smoke_test_per_class", 2))
    train = RemappedCIFARSubset(base_train, classes, train_limit, seed)
    test = RemappedCIFARSubset(base_test, classes, test_limit, seed + 1)
    num_workers = int(cfg.get("num_workers", 8))
    loader_kwargs = {
        "batch_size": int(cfg.get("batch_size", 16)),
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(cfg.get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(cfg.get("prefetch_factor", 2))
    generator = torch.Generator().manual_seed(int(seed))
    per_epoch_limit = cfg.get("train_samples_per_class_per_epoch")
    if smoke_test and per_epoch_limit is not None:
        per_epoch_limit = min(int(per_epoch_limit), int(cfg.get("smoke_train_per_class", 4)))
    if per_epoch_limit is None:
        train_loader = DataLoader(train, shuffle=True, generator=generator, **loader_kwargs)
    else:
        train_loader = DataLoader(train, sampler=PerClassEpochSampler(train, per_epoch_limit, seed), shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test, shuffle=False, **loader_kwargs)
    return train_loader, test_loader, [base_train.classes[index] for index in classes]
