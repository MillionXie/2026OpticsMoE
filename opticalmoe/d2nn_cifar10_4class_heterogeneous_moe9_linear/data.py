import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import datasets, transforms

from utils import resolve_path


class PILToFloatTensor:
    def __call__(self, image):
        image = image.convert("L"); width, height = image.size
        value = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).view(height, width).unsqueeze(0)
        return value.float().div_(255.0)


class RemappedCIFARSubset(Dataset):
    def __init__(self, dataset, class_indices, samples_per_class=None, seed=7):
        self.dataset = dataset
        self.class_indices = [int(v) for v in class_indices]
        self.label_map = {value: index for index, value in enumerate(self.class_indices)}
        targets = torch.as_tensor(dataset.targets)
        generator = torch.Generator().manual_seed(int(seed))
        self.indices = []
        for class_index in self.class_indices:
            indices = (targets == class_index).nonzero().flatten()
            if samples_per_class is not None:
                order = torch.randperm(len(indices), generator=generator)
                indices = indices[order[: min(int(samples_per_class), len(indices))]]
            self.indices.extend(indices.tolist())
        order = torch.randperm(len(self.indices), generator=generator).tolist()
        self.indices = [self.indices[index] for index in order]
        selected_targets = targets[self.indices]
        self.targets = [self.label_map[int(value)] for value in selected_targets.tolist()]
        self.class_counts = {
            self.label_map[class_index]: int((selected_targets == class_index).sum().item())
            for class_index in self.class_indices
        }

    def __len__(self): return len(self.indices)

    def __getitem__(self, index):
        image, label = self.dataset[self.indices[index]]
        return image, self.label_map[int(label)]


class PerClassEpochSampler(Sampler):
    """Rotate through a capped number of samples per class each epoch.

    The underlying dataset remains complete. Within each class, consecutive
    epochs consume non-overlapping slices of a deterministic shuffled cycle;
    the selected cross-class indices are then shuffled together so mini-batches
    are not class-contiguous.
    """

    def __init__(self, dataset, samples_per_class, seed=7):
        self.samples_per_class = int(samples_per_class)
        if self.samples_per_class <= 0:
            raise ValueError("train_samples_per_class_per_epoch must be positive or null")
        self.seed = int(seed); self.epoch = 0
        labels = torch.as_tensor(dataset.targets)
        self.groups = {int(label): (labels == label).nonzero().flatten() for label in labels.unique(sorted=True)}

    def __len__(self):
        return sum(min(self.samples_per_class, len(indices)) for indices in self.groups.values())

    def _permutation(self, label, cycle):
        generator = torch.Generator().manual_seed(self.seed + int(label) * 1009 + int(cycle) * 100003)
        group = self.groups[label]
        return group[torch.randperm(len(group), generator=generator)]

    def __iter__(self):
        selected = []
        for label, group in self.groups.items():
            count = min(self.samples_per_class, len(group))
            offset = self.epoch * count; cycle, start = divmod(offset, len(group))
            remaining = count
            while remaining:
                permutation = self._permutation(label, cycle)
                take = min(remaining, len(group) - start)
                selected.extend(permutation[start:start + take].tolist())
                remaining -= take; cycle += 1; start = 0
        generator = torch.Generator().manual_seed(self.seed + self.epoch * 104729 + 17)
        order = torch.randperm(len(selected), generator=generator).tolist()
        self.epoch += 1
        return iter([selected[index] for index in order])


def create_loaders(config, seed=7, smoke_test=False):
    cfg = config.get("dataset", {})
    image_size = int(cfg.get("image_size", 100)); input_size = int(cfg.get("input_size", 120))
    if input_size < image_size or (input_size-image_size) % 2 != 0: raise ValueError("input_size-image_size must be nonnegative and even.")
    pad = (input_size-image_size)//2
    interpolation_name=str(cfg.get("resize_interpolation","bicubic")).lower()
    interpolation_modes={
        "nearest":transforms.InterpolationMode.NEAREST,
        "bilinear":transforms.InterpolationMode.BILINEAR,
        "bicubic":transforms.InterpolationMode.BICUBIC,
    }
    if interpolation_name not in interpolation_modes:
        raise ValueError("dataset.resize_interpolation must be nearest, bilinear, or bicubic.")
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(
            (image_size,image_size),interpolation=interpolation_modes[interpolation_name],
            antialias=bool(cfg.get("resize_antialias",True)),
        ),
        transforms.Pad(pad, fill=0, padding_mode="constant"),
        PILToFloatTensor(),
    ])
    root = resolve_path(cfg.get("root", "./data"))
    train = datasets.CIFAR10(str(root), train=True, download=bool(cfg.get("download", True)), transform=transform)
    test = datasets.CIFAR10(str(root), train=False, download=bool(cfg.get("download", True)), transform=transform)
    base_class_names = list(train.classes)
    classes = cfg.get("class_indices", [0, 1, 2, 3])
    train_limit = cfg.get("train_samples_per_class")
    test_limit = cfg.get("test_samples_per_class")
    if smoke_test:
        train_limit = int(cfg.get("smoke_train_per_class", 4)); test_limit = int(cfg.get("smoke_test_per_class", 2))
    train = RemappedCIFARSubset(train, classes, train_limit, seed)
    test = RemappedCIFARSubset(test, classes, test_limit, seed+1)
    generator = torch.Generator().manual_seed(seed)
    num_workers = int(cfg.get("num_workers", 4))
    kwargs = {
        "batch_size": int(cfg.get("batch_size", 16)),
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    # These options only reduce host-side loading overhead.  They do not drop
    # samples: shuffle=True still visits every selected training sample once
    # per epoch, split into mini-batches of dataset.batch_size.
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(cfg.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(cfg.get("prefetch_factor", 2))
    per_epoch_limit = cfg.get("train_samples_per_class_per_epoch")
    if smoke_test and per_epoch_limit is not None:
        per_epoch_limit = min(int(per_epoch_limit), int(cfg.get("smoke_train_per_class", 4)))
    if per_epoch_limit is None:
        train_loader = DataLoader(train, shuffle=True, generator=generator, **kwargs)
    else:
        train_loader = DataLoader(train, sampler=PerClassEpochSampler(train, per_epoch_limit, seed), shuffle=False, **kwargs)
    test_loader = DataLoader(test, shuffle=False, **kwargs)
    class_names = [base_class_names[index] for index in classes]
    return train_loader, test_loader, class_names
