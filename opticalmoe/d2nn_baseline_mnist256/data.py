import torch
from torch.utils.data import DataLoader, Subset
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


def mnist_transform(input_size):
    return transforms.Compose(
        [
            transforms.Resize((int(input_size), int(input_size))),
            PILToFloatTensorNoNumpy(),
        ]
    )


def create_mnist_loaders(config, seed=7, smoke_test=False):
    dataset_cfg = config.get("dataset", {})
    root = resolve_path(dataset_cfg.get("root", "./data"))
    input_size = int(dataset_cfg.get("input_size", 256))
    batch_size = int(dataset_cfg.get("batch_size", 128))
    num_workers = int(dataset_cfg.get("num_workers", 0))
    download = bool(dataset_cfg.get("download", True))
    transform = mnist_transform(input_size)
    train_set = datasets.MNIST(root=str(root), train=True, download=download, transform=transform)
    test_set = datasets.MNIST(root=str(root), train=False, download=download, transform=transform)
    if smoke_test:
        train_set = Subset(train_set, range(min(int(dataset_cfg.get("smoke_train_size", 256)), len(train_set))))
        test_set = Subset(test_set, range(min(int(dataset_cfg.get("smoke_test_size", 128)), len(test_set))))
    generator = torch.Generator().manual_seed(int(seed))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available(), generator=generator)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    class_names = [str(i) for i in range(10)]
    return train_loader, test_loader, class_names

