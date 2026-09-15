from pathlib import Path
from types import SimpleNamespace
import io
import tarfile

import torch
from PIL import Image

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.cache_teacher_embeddings import (
    cache_identity,
)
from experiments.qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval.categories import (
    CALTECH101_CATEGORIES,
)
from experiments.qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval.prepare_caltech101_retrieval import (
    BACKGROUND_CATEGORY,
    _ensure_dataset,
    prepare_caltech101_subset,
)
from experiments.qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval.settings import (
    load_settings,
)
from experiments.qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval.teacher_cache import (
    _derive_subset_cache,
)


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
CONFIG = EXPERIMENT_DIR / "configs" / "release" / "caltech101_robust_hybrid_moe4.yaml"
TARGET10_CONFIG = (
    EXPERIMENT_DIR / "configs" / "release" / "caltech101_target10_finetune.yaml"
)


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
    assert settings.teacher_cache_path.parent == (
        EXPERIMENT_DIR / "cache" / "caltech101_all101_seed42_g3_train30_test20"
    ).resolve()


def test_target10_resumes_with_lower_lrs_and_shared_subset_cache() -> None:
    settings = load_settings(TARGET10_CONFIG)
    assert settings.selected_skus == (
        "airplanes",
        "Motorbikes",
        "Faces",
        "Leopards",
        "accordion",
        "grand_piano",
        "scorpion",
        "sunflower",
        "watch",
        "yin_yang",
    )
    assert settings.epochs == 20
    assert settings.learning_rate == 5.0e-5
    assert settings.phase_learning_rate == 1.0e-5
    assert settings.evaluate_test_each_epoch
    assert settings.teacher_cache_path.parent.name.startswith("caltech101_target10")
    assert settings.teacher_cache_source_path is not None
    assert settings.teacher_cache_source_path.parent.name.startswith(
        "caltech101_all101"
    )


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


def test_existing_inner_tar_is_extracted_before_network_download(tmp_path: Path) -> None:
    archive = tmp_path / "caltech-101/101_ObjectCategories.tar.gz"
    archive.parent.mkdir(parents=True)
    payload = b"offline-image"
    with tarfile.open(archive, "w:gz") as target:
        info = tarfile.TarInfo("101_ObjectCategories/airplanes/image_0001.jpg")
        info.size = len(payload)
        target.addfile(info, io.BytesIO(payload))
    settings = SimpleNamespace(
        dataset_root=tmp_path,
        download=False,
    )

    categories = _ensure_dataset(settings)

    assert categories == archive.parent / "101_ObjectCategories"
    assert (categories / "airplanes/image_0001.jpg").read_bytes() == payload


def test_target_cache_is_sliced_from_all_class_cache_without_teacher_forward(
    tmp_path: Path,
) -> None:
    categories_root = tmp_path / "101_ObjectCategories"
    for category in ("alpha", "beta", "gamma"):
        directory = categories_root / category
        directory.mkdir(parents=True)
        for index in range(8):
            Image.new("RGB", (4, 4), color=(index, 0, 0)).save(
                directory / f"image_{index:04d}.jpg"
            )

    def make_settings(selected: tuple[str, ...], variant: str) -> SimpleNamespace:
        return SimpleNamespace(
            dataset_root=tmp_path,
            download=False,
            selected_skus=selected,
            gallery_images_per_sku=2,
            train_limit_per_sku=3,
            test_limit_per_sku=2,
            random_seed=42,
            subset_manifest_path=tmp_path / f"{variant}.csv",
            output_dir=tmp_path / variant,
            dataset_variant=variant,
            model_id="teacher",
            instruction="same instruction",
            processor_min_pixels=16,
            processor_max_pixels=16,
            embedding_dim=64,
        )

    source_settings = make_settings(("alpha", "beta", "gamma"), "all")
    source_bundle = prepare_caltech101_subset(source_settings, persist=False)
    source_path = tmp_path / "all_cache" / "teacher_embeddings.pt"
    source_path.parent.mkdir()
    embeddings = torch.nn.functional.normalize(
        torch.randn(len(source_bundle.all_samples()), 64), dim=-1
    ).half()
    torch.save(
        {
            "metadata": cache_identity(source_bundle, source_settings),
            "records": [sample.manifest_record() for sample in source_bundle.all_samples()],
            "teacher_embeddings": embeddings,
        },
        source_path,
    )
    target_settings = make_settings(("alpha", "gamma"), "target")
    target_bundle = prepare_caltech101_subset(target_settings, persist=False)
    destination = tmp_path / "target_cache" / "teacher_embeddings.pt"
    result = _derive_subset_cache(
        source_path, destination, target_bundle, target_settings
    )
    payload = torch.load(result, map_location="cpu", weights_only=False)
    assert payload["metadata"]["derived_without_teacher_forward"] is True
    assert payload["teacher_embeddings"].shape == (
        len(target_bundle.all_samples()),
        64,
    )
