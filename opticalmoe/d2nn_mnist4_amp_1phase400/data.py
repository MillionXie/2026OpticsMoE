import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from utils import resolve_path


class PILToFloatTensorNoNumpy:
    """Convert PIL image to [1,H,W] float tensor without torchvision ToTensor."""

    def __call__(self, image):
        image = image.convert("L")
        width, height = image.size
        tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        tensor = tensor.view(height, width).unsqueeze(0)
        return tensor.to(dtype=torch.float32).div_(255.0)


def mnist_transform(dataset_cfg):
    """Build either the notebook-style resize+pad path or direct resize."""
    mode = str(dataset_cfg.get("preprocess_mode", "resize_then_pad"))
    output_size = int(dataset_cfg.get("input_size", 400))
    resize_size = int(dataset_cfg.get("resize_size", 336))
    interpolation_name = str(dataset_cfg.get("interpolation", "bicubic")).lower()
    interpolation = {
        "nearest": transforms.InterpolationMode.NEAREST,
        "bilinear": transforms.InterpolationMode.BILINEAR,
        "bicubic": transforms.InterpolationMode.BICUBIC,
    }.get(interpolation_name)
    if interpolation is None:
        raise ValueError(f"Unsupported dataset.interpolation: {interpolation_name}")
    if mode == "resize_then_pad":
        if resize_size > output_size:
            raise ValueError("dataset.resize_size must be <= dataset.input_size for resize_then_pad.")
        difference = output_size - resize_size
        left = difference // 2
        right = difference - left
        return transforms.Compose(
            [
                transforms.Resize((resize_size, resize_size), interpolation=interpolation),
                transforms.Pad((left, left, right, right), fill=0, padding_mode="constant"),
                PILToFloatTensorNoNumpy(),
            ]
        )
    if mode == "direct_resize":
        return transforms.Compose(
            [
                transforms.Resize((output_size, output_size), interpolation=interpolation),
                PILToFloatTensorNoNumpy(),
            ]
        )
    raise ValueError(f"Unsupported dataset.preprocess_mode: {mode}")


class RemappedDigitSubset(Dataset):
    def __init__(self, dataset, digits, samples_per_class=None, seed=7):
        self.dataset = dataset
        self.digits = [int(item) for item in digits]
        self.label_map = {digit: index for index, digit in enumerate(self.digits)}
        targets = torch.as_tensor(dataset.targets)
        generator = torch.Generator().manual_seed(int(seed))
        self.indices = []
        for digit in self.digits:
            digit_indices = (targets == digit).nonzero(as_tuple=False).flatten()
            if samples_per_class is not None:
                requested = int(samples_per_class)
                if requested <= 0:
                    raise ValueError("samples_per_class must be positive or null.")
                order = torch.randperm(len(digit_indices), generator=generator)
                digit_indices = digit_indices[order[: min(requested, len(digit_indices))]]
            self.indices.extend(digit_indices.tolist())
        if self.indices:
            order = torch.randperm(len(self.indices), generator=generator).tolist()
            self.indices = [self.indices[index] for index in order]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, target = self.dataset[self.indices[index]]
        return image, self.label_map[int(target)]


def create_mnist_loaders(config, seed=7, smoke_test=False):
    dataset_cfg = config.get("dataset", {})
    root = resolve_path(dataset_cfg.get("root", "./data"))
    batch_size = int(dataset_cfg.get("batch_size", 128))
    num_workers = int(dataset_cfg.get("num_workers", 0))
    download = bool(dataset_cfg.get("download", True))
    transform = mnist_transform(dataset_cfg)
    train_set = datasets.MNIST(root=str(root), train=True, download=download, transform=transform)
    test_set = datasets.MNIST(root=str(root), train=False, download=download, transform=transform)
    digits = dataset_cfg.get("class_digits", list(range(10)))
    use_full_dataset = bool(dataset_cfg.get("use_full_dataset", False))
    train_per_class = None if use_full_dataset else dataset_cfg.get("train_samples_per_class", 3000)
    test_per_class = None if use_full_dataset else dataset_cfg.get("test_samples_per_class", 600)
    if smoke_test:
        train_per_class = max(1, int(dataset_cfg.get("smoke_train_size", 32)) // len(digits))
        test_per_class = max(1, int(dataset_cfg.get("smoke_test_size", 16)) // len(digits))
    train_set = RemappedDigitSubset(train_set, digits, train_per_class, seed=seed)
    test_set = RemappedDigitSubset(test_set, digits, test_per_class, seed=int(seed) + 1)
    generator = torch.Generator().manual_seed(int(seed))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available(), generator=generator)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    class_names = [str(i) for i in digits]
    return train_loader, test_loader, class_names
