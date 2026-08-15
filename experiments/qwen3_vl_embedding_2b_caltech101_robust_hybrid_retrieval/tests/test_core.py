from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from experiments.qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval.categories import (
    CALTECH101_CATEGORIES,
)
from experiments.qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval.prepare_caltech101_retrieval import (
    BACKGROUND_CATEGORY,
    prepare_caltech101_subset,
)
from experiments.qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval.settings import (
    load_settings,
)


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
CONFIG = EXPERIMENT_DIR / "configs" / "release" / "caltech101_robust_hybrid_moe4.yaml"


def test_release_config_uses_all_categories_and_normal_lrs() -> None:
    settings = load_settings(CONFIG)
    assert len(CALTECH101_CATEGORIES) == 101
    assert settings.selected_skus == CALTECH101_CATEGORIES
    assert settings.dataset_variant == "caltech101_101class"
    assert settings.gallery_images_per_sku == 3
    assert settings.train_limit_per_sku == 30
    assert settings.test_limit_per_sku == 20
    assert settings.optimizer_steps_per_epoch is None
    assert settings.learning_rate == 1.0e-4
    assert settings.phase_learning_rate == 2.0e-5
    assert not settings.evaluate_test_each_epoch


def test_seeded_caltech_split_is_capped_disjoint_and_reproducible(tmp_path: Path) -> None:
    categories_root = tmp_path / "101_ObjectCategories"
    selected = ("alpha", "beta")
    for category in (*selected, BACKGROUND_CATEGORY):
        directory = categories_root / category
        directory.mkdir(parents=True)
        for index in range(8):
            Image.new("RGB", (4, 4), color=(index, 0, 0)).save(
                directory / f"image_{index:04d}.jpg"
            )
    output = tmp_path / "run"
    settings = SimpleNamespace(
        dataset_root=tmp_path,
        download=False,
        selected_skus=selected,
        gallery_images_per_sku=2,
        train_limit_per_sku=3,
        test_limit_per_sku=2,
        random_seed=42,
        subset_manifest_path=output / "manifests" / "subset.csv",
        output_dir=output,
        dataset_variant="test",
    )
    first = prepare_caltech101_subset(settings, persist=False)
    second = prepare_caltech101_subset(settings, persist=False)
    assert (len(first.gallery_samples), len(first.train_samples), len(first.test_samples)) == (
        4,
        6,
        4,
    )
    paths = [
        {sample.image_path for sample in samples}
        for samples in (first.gallery_samples, first.train_samples, first.test_samples)
    ]
    assert paths[0].isdisjoint(paths[1])
    assert paths[0].isdisjoint(paths[2])
    assert paths[1].isdisjoint(paths[2])
    assert first.manifest_digest == second.manifest_digest
    assert BACKGROUND_CATEGORY not in first.class_names
