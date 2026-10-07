import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset
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


class MNISTWithClipFeatures(Dataset):
    """MNIST dataset that pairs images with pre-extracted CLIP features by index."""

    def __init__(self, mnist_dataset, clip_features, clip_labels):
        self.mnist = mnist_dataset
        self.features = clip_features.astype(np.float32)
        self.feature_labels = clip_labels.astype(np.int64)
        # Build index mapping: label -> feature index (features may be in original order)
        # Since MNIST and CLIP features are saved in the same order, use direct indexing
        self._indices = None

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        image, label = self.mnist[idx]
        feat = torch.from_numpy(self.features[idx])
        return image, feat, label


def create_mnist_clip_loaders(config, seed=7, smoke_test=False):
    """Create MNIST loaders paired with pre-extracted CLIP features.

    Config keys used:
      dataset.root, dataset.input_size, dataset.batch_size, dataset.num_workers,
      dataset.clip_feature_dir
    """
    dataset_cfg = config.get("dataset", {})
    root = resolve_path(dataset_cfg.get("root", "./data"))
    input_size = int(dataset_cfg.get("input_size", 256))
    batch_size = int(dataset_cfg.get("batch_size", 2500))
    num_workers = int(dataset_cfg.get("num_workers", 0))
    download = bool(dataset_cfg.get("download", True))
    clip_dir = resolve_path(dataset_cfg.get("clip_feature_dir", "./data/mnist_clip_features"))

    # Standard MNIST transform
    transform = mnist_transform(input_size)

    # Load raw MNIST
    train_set_raw = datasets.MNIST(root=str(root), train=True, download=download, transform=transform)
    test_set_raw = datasets.MNIST(root=str(root), train=False, download=download, transform=transform)

    # Load CLIP features
    train_feat = np.load(os.path.join(clip_dir, "train_clip_features.npy"))
    train_lbl = np.load(os.path.join(clip_dir, "train_labels.npy"))
    test_feat = np.load(os.path.join(clip_dir, "test_clip_features.npy"))
    test_lbl = np.load(os.path.join(clip_dir, "test_labels.npy"))

    # Wrap with CLIP features
    train_set = MNISTWithClipFeatures(train_set_raw, train_feat, train_lbl)
    test_set = MNISTWithClipFeatures(test_set_raw, test_feat, test_lbl)

    if smoke_test:
        train_set = Subset(train_set, range(min(int(dataset_cfg.get("smoke_train_size", 256)), len(train_set))))
        test_set = Subset(test_set, range(min(int(dataset_cfg.get("smoke_test_size", 128)), len(test_set))))

    generator = torch.Generator().manual_seed(int(seed))
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    class_names = [str(i) for i in range(10)]
    return train_loader, test_loader, class_names


# ==============================================================================
#  CIFAR-10 loaders
# ==============================================================================

class CIFAR10ToGrayTensor:
    def __call__(self, image):
        image = image.convert("L").resize((256, 256))
        tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        return tensor.view(256, 256).unsqueeze(0).to(dtype=torch.float32).div_(255.0)


def create_cifar10_loaders(config, seed=7, smoke_test=False):
    dataset_cfg = config.get("dataset", {})
    root = resolve_path(dataset_cfg.get("root", "./data"))
    batch_size = int(dataset_cfg.get("batch_size", 256))
    num_workers = int(dataset_cfg.get("num_workers", 8))
    transform = CIFAR10ToGrayTensor()
    train_set = datasets.CIFAR10(root=str(root), train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root=str(root), train=False, download=True, transform=transform)
    if smoke_test:
        train_set = Subset(train_set, range(min(int(dataset_cfg.get("smoke_train_size", 256)), len(train_set))))
        test_set = Subset(test_set, range(min(int(dataset_cfg.get("smoke_test_size", 128)), len(test_set))))
    gen = torch.Generator().manual_seed(int(seed))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True, generator=gen)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, ['airplane','auto','bird','cat','deer','dog','frog','horse','ship','truck']


class CIFAR10WithCLIP(Dataset):
    """CIFAR-10 + on-the-fly CLIP RN50 feature extraction."""
    def __init__(self, cifar10_ds, clip_model, clip_preprocess, device):
        from PIL import Image
        self.ds = cifar10_ds
        self.clip = clip_model
        self.preprocess = clip_preprocess
        self.device = device
    def __len__(self): return len(self.ds)
    @torch.no_grad()
    def __getitem__(self, idx):
        from PIL import Image
        img, label = self.ds[idx]
        pil = Image.fromarray((img.squeeze(0).numpy()*255).astype(np.uint8), 'L').convert('RGB')
        clip_in = self.preprocess(pil).unsqueeze(0).to(self.device)
        feat = self.clip.encode_image(clip_in).squeeze(0).cpu().float()
        return img, feat, label


def create_cifar10_clip_loaders(config, seed=7, smoke_test=False):
    import clip
    from PIL import Image
    dataset_cfg = config.get("dataset", {})
    root = resolve_path(dataset_cfg.get("root", "./data"))
    batch_size = int(dataset_cfg.get("batch_size", 256))
    num_workers = 0  # must be 0 due to CUDA in subprocess
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load("RN50", device=device, jit=False)
    clip_model.eval()
    transform = CIFAR10ToGrayTensor()
    train_base = datasets.CIFAR10(root=str(root), train=True, download=True, transform=transform)
    test_base = datasets.CIFAR10(root=str(root), train=False, download=True, transform=transform)
    if smoke_test:
        train_base = Subset(train_base, range(min(int(dataset_cfg.get("smoke_train_size", 256)), len(train_base))))
        test_base = Subset(test_base, range(min(int(dataset_cfg.get("smoke_test_size", 128)), len(test_base))))
    train_set = CIFAR10WithCLIP(train_base, clip_model, clip_preprocess, device)
    test_set = CIFAR10WithCLIP(test_base, clip_model, clip_preprocess, device)
    gen = torch.Generator().manual_seed(int(seed))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True, generator=gen)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, ['airplane','auto','bird','cat','deer','dog','frog','horse','ship','truck']

