import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.data import datasets as data_mod


class TinyUSPS:
    classes = [str(i) for i in range(10)]

    def __init__(self, root, train=True, transform=None, download=False):
        self.train = train
        self.transform = transform
        count = 30 if train else 20
        self.targets = [idx % 10 for idx in range(count)]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = Image.new("L", (16, 16), color=int(index) % 255)
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.targets[index])


def test_usps_dataset_registry_shape_label_sampling_and_smoke(monkeypatch, tmp_path):
    monkeypatch.setitem(data_mod.DATASET_REGISTRY, "usps", TinyUSPS)
    bundle = data_mod.create_dataloaders(
        {
            "name": "usps",
            "root": str(tmp_path),
            "input_size": 134,
            "grayscale": True,
            "batch_size": 4,
            "num_workers": 0,
            "download": True,
            "val_split": 0.1,
            "sampling_protocol": {
                "enabled": True,
                "total_size": 20,
                "train_test_ratio": [4, 1],
                "class_balanced": True,
            },
            "smoke_test": True,
            "smoke_train_size": 8,
            "smoke_test_size": 4,
        },
        seed=7,
    )
    images, labels = next(iter(bundle.train_loader))
    assert images.shape[1:] == (1, 134, 134)
    assert labels.dtype.is_floating_point is False
    assert int(labels.min()) >= 0
    assert int(labels.max()) <= 9
    assert bundle.num_classes == 10
    assert len(bundle.test_loader.dataset) == 4

