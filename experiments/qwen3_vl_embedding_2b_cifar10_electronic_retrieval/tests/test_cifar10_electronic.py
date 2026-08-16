from __future__ import annotations

from pathlib import Path

from PIL import Image

from experiments.qwen3_vl_embedding_2b_cifar10_electronic_retrieval.prepare_cifar10_retrieval import (
    _materialize_samples,
)
from experiments.qwen3_vl_embedding_2b_cifar10_electronic_retrieval.settings import (
    CIFAR10_CLASSES,
    load_settings,
)


class _TinyCIFAR:
    def __getitem__(self, index: int):
        return Image.new("RGB", (32, 32), (index, 20, 30)), 0


def test_release_config_uses_full_official_train_without_teacher() -> None:
    settings = load_settings(
        "experiments/qwen3_vl_embedding_2b_cifar10_electronic_retrieval/"
        "configs/release/cifar10_electronic_vision2d.yaml"
    )
    assert tuple(settings.selected_skus) == CIFAR10_CLASSES
    assert settings.train_limit_per_sku is None
    assert settings.test_limit_per_sku is None
    assert settings.gallery_images_per_sku == 3
    assert settings.epochs == 12
    assert settings.teacher_enabled is False
    assert settings.electronic_deepstack_enabled is False
    assert settings.electronic_vision_token_mixer_type == "depthwise_conv2d"
    assert settings.electronic_language_token_mixer_type == "depthwise_conv1d"


def test_cifar_materialization_produces_stable_grocery_samples(tmp_path: Path) -> None:
    samples = _materialize_samples(
        _TinyCIFAR(),
        [0, 1],
        0,
        "airplane",
        "train",
        "official_train",
        tmp_path,
        False,
    )
    assert [sample.sample_id for sample in samples] == [
        "cifar10:official_train:00000",
        "cifar10:official_train:00001",
    ]
    assert all(sample.image_path.is_file() for sample in samples)
    assert all(sample.sku_name == "airplane" for sample in samples)
